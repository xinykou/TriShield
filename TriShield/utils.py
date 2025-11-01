from typing import Callable, List, Tuple
import torch
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer
import contextlib
import functools
from copy import deepcopy


def formatInp_llama_persuasion(
    model_name,
    tokenizer,
    prompts: List[str],
):
    if model_name == "qwen3" or model_name == "qwen2":
        template = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant"
    elif model_name == "llama3" or model_name == "llama3.2":
        template = "<|start_header_id|>user<|end_header_id|>\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    elif model_name == "innospark":
        template = "<|im_start|>system\nYou are InnoSpark, created by Lab of AI Education. You are from East China Normal University(华东师范大学), and your Chinese Name is 启创. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}.<|im_end|>\n<|im_start|>assistant"
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    start_positions = []
    tokenized_batch = tokenizer(
        [template.format(p) for p in prompts],
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    max_len = tokenized_batch.input_ids.shape[1]

    for i, _ in enumerate(prompts):
        # 单独 encode 模板（只到 prompt 前）
        template_prefix = template.split("{}")[0]
        template_ids = tokenizer(template_prefix, add_special_tokens=False).input_ids
        template_len = len(template_ids)

        # batch 内该序列真实长度（不含左 padding）
        seq_len = (tokenized_batch.attention_mask[i] == 1).sum().item()

        # 左 padding 数量
        pad_len = max_len - seq_len

        # prompt 在这个序列中的起始位置
        start_pos = pad_len + template_len
        start_positions.append(start_pos)

        print()
    return tokenized_batch, start_positions


# 用于标记是否已经触发过 hook
embed_tokens_hook_triggered_id = -1


def get_highlighted_embed_hook(
    batch_start_indices=None,
    batch_offset_indices=None,
    batch_id=None,
    removal_strength=1.0,
):
    def hook_fn(
        module: torch.nn.Module, input: Tuple[torch.Tensor, ...], output: torch.Tensor
    ):
        global embed_tokens_hook_triggered_id
        rs = torch.tensor(
            removal_strength, device=module.weight.device, dtype=output.dtype
        )

        if embed_tokens_hook_triggered_id != batch_id:
            embed_tokens_hook_triggered_id = batch_id
            # 只在需要时 clone
            new_embeds = output.clone()

            for i, token_indices in enumerate(batch_offset_indices):
                # 一次性缩放
                positions = torch.tensor(
                    token_indices, device=new_embeds.device, dtype=torch.long
                )
                new_embeds[i, positions] *= rs

            return new_embeds
        else:
            return output

    return hook_fn


def get_refusal_or_not_fwd_hook(
    deltas: Tensor,
    harmful_indexs,
) -> Callable:
    def hook_fn(
        module: torch.nn.Module, input: Tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> Tensor:
        if any(harmful_indexs):
            m = deepcopy(module)
            m.weight += deltas.to(m.weight.device)
            current_output = m.forward(input[0])
            # print(f"current_output size: {current_output.shape}")
            # print(f"output size: {output.shape}")
            current_output[harmful_indexs == False] = output[harmful_indexs == False]
            return current_output
        else:
            print("no harmful in this batch")
            return output

    return hook_fn


def get_mean_activations_pre_hook(
    layer: int,
    cache_full: List[List[Tensor]],
    positions: List[int],
    whole_seq: bool = False,
    step: int = None,
) -> Callable:
    """
    Creates a hook function to collect mean activations.

    Args:
        layer: Layer number
        cache_full: Cache to store activations
        positions: Positions to extract activations from
        whole_seq: Whether to store whole sequence
        step: Number of tokens to consider

    Returns:
        Hook function that collects activations
    """

    def hook_fn(module: torch.nn.Module, input: Tuple[Tensor, ...]) -> None:
        activation = input[0].half()
        seq_len = activation.shape[1]

        if whole_seq:
            cache_full[layer].append(activation.clone().detach().cpu())
        else:
            if seq_len >= len(positions):
                # print("extracting positions", positions)
                assert isinstance(positions[0], int)
                # context = activation[:, -len(positions) - step : -len(positions), :]
                pos_activations = activation[:, positions, :]
                # print("extracting positions", positions, pos_activations.shape)
                # merged_activation = torch.cat([context, pos_activations], dim=1)
                # cache_full[layer].append(merged_activation.clone().detach().cpu())
                cache_full[layer].append(pos_activations.clone().detach().cpu())
            else:
                print("seq_len<positions", seq_len, len(positions))
                exit()

    return hook_fn


def get_mean_activations_fwd_hook(
    layer: int,
    cache_full: List[List[Tensor]],
    positions: List[int],
    whole_seq: bool = False,
    step: int = None,
) -> Callable:
    """
    Creates a forward hook function to collect mean activations.

    Args:
        layer: Layer number
        cache_full: Cache to store activations
        positions: Positions to extract activations from
        whole_seq: Whether to store whole sequence
        step: Number of tokens to consider

    Returns:
        Hook function that collects activations
    """

    def hook_fn(
        module: torch.nn.Module, input: Tuple[Tensor, ...], output: Tuple[Tensor, ...]
    ) -> None:
        activation = output.half() if output.dim() == 3 else output[0].half()
        seq_len = activation.shape[1]

        if whole_seq:
            cache_full[layer].append(activation.clone().detach().cpu())
        else:
            if seq_len >= len(positions):
                # context = activation[:, -len(positions) - step : -len(positions), :]
                pos_activations = activation[:, positions, :]
                # merged_activation = torch.cat([context, pos_activations], dim=1)
                # cache_full[layer].append(merged_activation.clone().detach().cpu())
                cache_full[layer].append(pos_activations.clone().detach().cpu())
            else:
                print("seq_len<positions", seq_len, len(positions))
                exit()

    return hook_fn


def get_mean_activations_fwd_hook(
    layer: int,
    cache_full: List[List[Tensor]],
    positions: List[int],
    whole_seq: bool = False,
    step: int = None,
) -> Callable:
    """
    Creates a forward hook function to collect mean activations.

    Args:
        layer: Layer number
        cache_full: Cache to store activations
        positions: Positions to extract activations from
        whole_seq: Whether to store whole sequence
        step: Number of tokens to consider

    Returns:
        Hook function that collects activations
    """

    def hook_fn(
        module: torch.nn.Module, input: Tuple[Tensor, ...], output: Tuple[Tensor, ...]
    ) -> None:
        activation = output.half() if output.dim() == 3 else output[0].half()
        seq_len = activation.shape[1]

        if whole_seq:
            cache_full[layer].append(activation.clone().detach().cpu())
        else:
            if seq_len >= len(positions):
                # context = activation[:, -len(positions) - step : -len(positions), :]
                pos_activations = activation[:, positions, :]
                # merged_activation = torch.cat([context, pos_activations], dim=1)
                # cache_full[layer].append(merged_activation.clone().detach().cpu())
                cache_full[layer].append(pos_activations.clone().detach().cpu())
            else:
                print("seq_len<positions", seq_len, len(positions))
                exit()

    return hook_fn


@contextlib.contextmanager
def add_hooks(
    module_forward_pre_hooks: List[Tuple[torch.nn.Module, Callable]],
    module_forward_hooks: List[Tuple[torch.nn.Module, Callable]],
    **kwargs,
) -> None:
    """
    Context manager for temporarily adding forward hooks to a model.

    Args:
        module_forward_pre_hooks: A list of pairs: (module, fnc) The function will be registered as a
            forward pre hook on the module
        module_forward_hooks: A list of pairs: (module, fnc) The function will be registered as a
            forward hook on the module
        **kwargs: Additional keyword arguments to pass to the hooks
    """
    try:
        handles = []
        for module, hook in module_forward_pre_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_pre_hook(partial_hook))

        for module, hook in module_forward_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_hook(partial_hook))
        yield
    finally:
        for h in handles:
            h.remove()
