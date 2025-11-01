import os
import json
import argparse

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import (
    formatInp_llama_persuasion,
    get_mean_activations_pre_hook,
    get_mean_activations_fwd_hook,
    get_highlighted_embed_hook,
    add_hooks,
)
from tqdm import tqdm
from peft import PeftModel
from th_llm import TokenHighlighter

NUM_TOKEN_HIDDEN = 2  # by default, we extract NUM_TOKEN_HIDDEN tokens + all special post-instruction tokens


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert JSON files to a specific format."
    )
    parser.add_argument("--model_name", default="qwen3", type=str, help="Model type")
    parser.add_argument("--model_path", default="/media/1/public_model/Qwen3-8B")
    parser.add_argument(
        "--lora_path",
        default="",
        type=str,
    )
    parser.add_argument(
        "--data_path",
        default="/media/4/yx/education_safety/panda-guard/result/jailbreak_attack/_media_1_public_model_llama-160m/AutoDanAttacker_AutoDan/NoneDefender/results.json",
        type=str,
        help="Path to harmful examples",
    )
    parser.add_argument(
        "--data_name",
        default="AutoDAN",
        type=str,
    )
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument(
        "--data_source",
        default="jailbreak",
        choices=["jailbreak"],
        help="Whether to run in test mode",
    )
    parser.add_argument(
        "--batch_size",
        default=8,
        type=int,
    )
    parser.add_argument(
        "--extract_harmful_token_only",
        action="store_false",
        help="Whether to extract only the harmful token or also the special post-instruction tokens",
    )
    parser.add_argument(
        "--output_dir",
        default="result/toy_attention/qwen3/Highlighter/jailbreak_attack",
    )
    parser.add_argument(
        "--removal_strength",
        default=1.0,
        type=float,
        help="Strength of removal for highlighted tokens",
    )
    parser.add_argument(
        "--removal_ratio",
        default=0.1,
        type=float,
        help="Ratio of tokens to remove based on highlighting",
    )
    args = parser.parse_args()

    model_name = args.model_name
    data_path = args.data_path
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, device_map="auto", torch_dtype=torch.bfloat16
    )
    if args.lora_path and os.path.exists(args.lora_path):
        print("Recover LoRA weights...")
        model = PeftModel.from_pretrained(model, args.lora_path)
        model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, use_fast=True, padding_side="left"
    )
    tokenizer.pad_token = tokenizer.eos_token
    if model_name == "qwen3" or "qwen2":
        inst_token = "<|im_end|>\n<|im_start|>assistant"
    elif model_name == "llama3":
        inst_token = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    tokenized_inst = tokenizer(
        inst_token, return_tensors="pt", add_special_tokens=False
    )
    if args.extract_harmful_token_only:  # inst
        positions = [-len(tokenized_inst.input_ids[0]) - 1]
        NUM_TOKEN_HIDDEN = 0
        pos_extraction_name = "inst"
    else:  # post-inst
        positions = [i for i in range(-len(tokenized_inst.input_ids[0]), 0, 1)]
        pos_extraction_name = "post_inst"

    with open(args.data_path, "r") as f:
        data = json.load(f)

    if args.data_source == "jailbreak":
        prompts = [item["result"][0]["messages"] for item in data["results"]]
        goals = [item["goal"] for item in data["results"]]
        data = [
            {"goal": goal, "prompt": prompt} for goal, prompt in zip(goals, prompts)
        ]
        if args.max_samples > 0:
            prompts = prompts[: args.max_samples]
    else:
        raise ValueError(f"Unsupported data source: {args.data_source}")

    # prompts = prompts[:8]  # for testing
    highter_index_all = []
    highter_unit_all = []
    ######## highlighter ##############
    segmenter = "en_core_web_trf"
    for prompt in tqdm(prompts, desc="Highlighter processing", total=len(prompts)):
        explainer = TokenHighlighter(
            model,
            "qwen3",
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
        highter_unit_all.append(output_dict["units"])

    block_modules = model.model.layers
    n_layers = model.config.num_hidden_layers
    full_activations = [[] for _ in range(n_layers + 1)]

    fwd_pre_hooks = [
        (
            block_modules[layer],
            get_mean_activations_pre_hook(
                layer=layer,
                cache_full=full_activations,
                positions=positions,
                step=NUM_TOKEN_HIDDEN,
            ),
        )
        for layer in range(n_layers)
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

    for i in tqdm(range(0, len(prompts), args.batch_size)):
        batch, start_positions = formatInp_llama_persuasion(
            model_name, tokenizer, prompts[i : i + args.batch_size]
        )
        inputs = batch.to(model.device)

        highter_index_batch = highter_index_all[i : i + args.batch_size]
        current_embed_hook = get_highlighted_embed_hook(
            batch_start_indices=start_positions,
            batch_offset_indices=highter_index_batch,
            batch_id=i,
            removal_strength=args.removal_strength,
        )
        dynamic_embed_hooks = [(model.model.embed_tokens, current_embed_hook)]
        with add_hooks(
            module_forward_pre_hooks=fwd_pre_hooks,
            module_forward_hooks=dynamic_embed_hooks + fwd_hooks,
        ):
            model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
            )
    # full_activations: layers, nums, tokens, hidden_dim
    flat_list = [
        torch.cat(inner_list) for inner_list in full_activations
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

    if args.data_source == "jailbreak":
        all_activations = result  # layers * nums * tokens * hidden_dim
        output_path = os.path.join(
            args.output_dir,
            f"highlighted_ratio_{args.removal_ratio}_strength_{args.removal_strength}_{args.data_name}-{pos_extraction_name}-test_hidden_states.pt",
        )
        torch.save(all_activations, output_path)
        print("all activations shape:", all_activations.shape)
    else:
        raise ValueError(f"Unsupported data mode: {args.data_source}")

    print(f"Hidden states saved to {output_path}")
