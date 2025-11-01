from typing import List
from tqdm import tqdm
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from utils import (
    add_hooks,
    get_mean_activations_pre_hook,
    get_mean_activations_fwd_hook,
    get_highlighted_embed_hook,
    formatInp_llama_persuasion,
)
from th_llm import TokenHighlighter
from torch import Tensor
import torch.nn.functional as F
import gc

NUM_TOKEN_HIDDEN = 2


def fuse_logits_average(current_logits: Tensor, temperature: float = 1.0):
    """
    使用 logit 平均进行融合 (公式化版本)
    Args:
        current_logits (Tensor): [num_layers, num_classes] 的 logits 张量
        temperature (float): 温度系数 (可选)，默认 1.0

    Returns:
        fused_probs (Tensor): [num_classes] 融合后的概率
        fused_pred  (Tensor): 融合后的预测类别索引
    """
    # N 表示层数
    N = current_logits.size(0)

    # 公式: (1/(N*T)) * sum_i L_i
    avg_logits = current_logits.sum(dim=0) / (N * temperature)

    # softmax 得到概率分布
    fused_probs = F.softmax(avg_logits, dim=-1)

    # 取概率最大的类别
    fused_pred = fused_probs.argmax(dim=-1)

    return fused_probs, fused_pred


def highlighter_init(
    args,
    model: PreTrainedModel,
    model_name: str,
    tokenizer: PreTrainedTokenizer,
    prompts: List[str],
):
    highter_index_all = []
    highter_unit_all = []
    for prompt in tqdm(prompts, desc="Highlighter processing", total=len(prompts)):
        try:
            explainer = TokenHighlighter(
                model,
                model_name,
                tokenizer,
                segmenter="en_core_web_trf",
                affirmation="Sure, I'd like to help you with this.",
                pooling="mean_norm",
            )

            output_dict = explainer.explain_instance(
                [prompt],
                unit_types=["p"],
                ind_segment=True,
                segment_type="token",  # split by sub
                max_phrase_length=10,
            )
            org_scores = output_dict["scores"]
            sorted_indices = sorted(
                range(len(org_scores)), key=lambda i: org_scores[i], reverse=True
            )
            # top 10% phrases
            nums = max(1, int(args.removal_ratio * len(sorted_indices)))
            top_indices = sorted_indices[:nums]
            highter_index_all.append(top_indices)  # pos for the current prompt
            # highter_unit_all.append(output_dict["units"])
            del explainer
            del output_dict
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            print(e)
            highter_index_all.append([])

    return highter_index_all


def highlghtered_or_not_extract_hidden(
    args,
    model: PreTrainedModel,
    model_name: str,
    tokenizer: PreTrainedTokenizer,
    prompts: List[str],
    positions: List[int],
    pos_extraction_name: str = "inst",
    highter_index_all: List[List[int]] = None,
    enable_highlighter: bool = True,
):

    block_modules = model.model.layers
    n_layers = model.config.num_hidden_layers
    full_activations = [[] for _ in range(n_layers + 1)]

    fwd_pre_hooks = [
        (
            block_modules[layer_id],
            get_mean_activations_pre_hook(
                layer=layer_id,
                cache_full=full_activations,
                positions=positions,
                step=NUM_TOKEN_HIDDEN,
            ),
        )
        for layer_id in range(n_layers)
    ]
    fwd_hooks = [
        (
            block_modules[n_layers - 1],
            get_mean_activations_fwd_hook(
                layer=-1,
                cache_full=full_activations,
                positions=positions,
                step=NUM_TOKEN_HIDDEN,
            ),
        )
    ]

    for i in tqdm(
        range(
            0,
            len(prompts),
            args.batch_size,
        ),
        total=len(prompts),
        desc="Extracting hidden states",
    ):
        templated_harmful_batch, start_positions = formatInp_llama_persuasion(
            model_name, tokenizer, prompts[i : i + args.batch_size]
        )
        inputs = templated_harmful_batch.to(model.device)

        if enable_highlighter:
            highter_index_batch = highter_index_all[i : i + args.batch_size]
            current_embed_hook = get_highlighted_embed_hook(
                batch_start_indices=start_positions,
                batch_offset_indices=highter_index_batch,
                batch_id=i,
                removal_strength=args.removal_strength,
            )
            dynamic_embed_hooks = [(model.model.embed_tokens, current_embed_hook)]
        else:
            dynamic_embed_hooks = []
        with add_hooks(
            module_forward_pre_hooks=fwd_pre_hooks,
            module_forward_hooks=(
                dynamic_embed_hooks + fwd_hooks if enable_highlighter else fwd_hooks
            ),
        ):
            with torch.no_grad():
                model(
                    input_ids=inputs.input_ids.to(model.device),
                    attention_mask=inputs.attention_mask.to(model.device),
                )
            # del inputs, templated_harmful_batch
            # torch.cuda.empty_cache()
            # gc.collect()

    # full_activations: layers * (nums, tokens, hidden_dim)
    flat_list = [
        torch.stack(inner_list) for inner_list in full_activations
    ]  # layers * tensor (nums, tokens, hidden_dim)
    result = torch.stack(flat_list).squeeze()  # (layers* nums * tokens* hidden_dim)

    if result.dim() == 2:
        result = result.unsqueeze(1).unsqueeze(2)  # layers * 1 * 1 * hidden_dim
    elif result.dim() == 3:
        if pos_extraction_name == "inst":
            result = result.unsqueeze(
                2
            )  # layers * nums * hidden_dim  --> layers * nums * 1 * hidden_dim
        else:
            result = result.unsqueeze(
                1
            )  # layers * tokens * hidden_dim ---> layers * 1 * tokens * hidden_dim
    elif result.dim() == 4:
        pass  # layers * nums * tokens * hidden_dim
    else:
        raise ValueError(f"Unexpected tensor shape: {result.shape}")

    return result  # layers * nums * tokens * hidden_dim
