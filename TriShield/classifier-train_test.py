import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
import json
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from peft import PeftModel
from classification_module.processs import (
    highlighter_init,
    highlghtered_or_not_extract_hidden,
    fuse_logits_average,
)
from classification_module.train import run_training
from classification_module.classifier_manager import load_classifier_manager

# by default, we extract NUM_TOKEN_HIDDEN tokens + all special post-instruction tokens


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
        "--data_paths",
        default=[
            "/media/4/yx/education_safety/result/toy/qwen3/accepted_harmless.json",
            "/media/4/yx/education_safety/result/toy/qwen3/refused_harmful.json",
        ],
        nargs="+",
        help="Path to refused harmful examples and accepted harmless examples",
    )
    parser.add_argument(
        "--data_source",
        default="vanilla",
    )
    parser.add_argument(
        "--data_mode",
        default="train",
        choices=["train", "test"],
        help="Whether to run in test mode",
    )
    parser.add_argument(
        "--not_extract_harmful_token_only",
        action="store_true",
    )
    parser.add_argument(
        "--batch_size",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--target_layer_ids",
        nargs="+",
        default=[5, 27, 29, 30, 34],
        type=int,
    )
    parser.add_argument(
        "--disable_highlighter",
        action="store_true",
    )
    parser.add_argument(
        "--removal_strength",
        type=float,
    )
    parser.add_argument(
        "--removal_ratio",
        type=float,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.0001,
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--output_dir",
        default="result/defense-jailbreak_attack/Classification_Module",
    )
    parser.add_argument("--max_samples", type=int, default=-1)
    args = parser.parse_args()

    model_name = args.model_name
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
    if model_name == "qwen3" or "qwen2" or model_name == "innospark":
        inst_token = "<|im_end|>\n<|im_start|>assistant"
    elif model_name == "llama3.2":
        inst_token = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    tokenized_inst = tokenizer(
        inst_token, return_tensors="pt", add_special_tokens=False
    )
    if not args.not_extract_harmful_token_only:  # inst
        positions = [-len(tokenized_inst.input_ids[0]) - 1]
        NUM_TOKEN_HIDDEN = 0
        pos_extraction_name = "inst"
    else:  # post-inst
        positions = [i for i in range(-len(tokenized_inst.input_ids[0]), 0, 1)]
        pos_extraction_name = "post_inst"

    if args.data_mode == "test":
        with open(args.data_paths[0], "r") as f:
            data = json.load(f)
        if args.data_source == "vanilla":
            prompts = [item["prompt"] for item in data]
            prompts = prompts[: args.max_samples] if args.max_samples > 0 else prompts
        elif "jailbreak" in args.data_source:
            prompts = [item["result"][0]["messages"] for item in data["results"]]
            prompts = prompts[: args.max_samples] if args.max_samples > 0 else prompts
    elif args.data_mode == "train":
        for current_pth in args.data_paths:
            with open(current_pth, "r") as f:
                data = json.load(f)
            if "harmful" in current_pth and "test" in current_pth:
                harmful_test_prompts = data
            elif "harmless" in current_pth and "test" in current_pth:
                harmless_test_prompts = data
            elif "harmful" in current_pth:
                harmful_prompts = data
            elif "harmless" in current_pth:
                harmless_prompts = data
    else:
        raise ValueError(
            f"Unsupported data source: {args.data_source} or data mode: {args.data_mode}"
        )

    if args.data_mode == "train":
        # Extract hidden states: (layer * nums * tokens * hidden_dim)
        harmful_hidden_group = highlghtered_or_not_extract_hidden(
            args,
            model,
            model_name,
            tokenizer,
            harmful_prompts,
            positions,
            pos_extraction_name=pos_extraction_name,
            highter_index_all=[],
            enable_highlighter=False,
        )
        harmless_hidden_group = highlghtered_or_not_extract_hidden(
            args,
            model,
            model_name,
            tokenizer,
            harmless_prompts,
            positions,
            pos_extraction_name=pos_extraction_name,
            highter_index_all=[],
            enable_highlighter=False,
        )

        harmful_test_hidden_group = highlghtered_or_not_extract_hidden(
            args,
            model,
            model_name,
            tokenizer,
            harmful_test_prompts,
            positions,
            pos_extraction_name=pos_extraction_name,
            highter_index_all=[],
            enable_highlighter=False,
        )

        harmless_test_hidden_group = highlghtered_or_not_extract_hidden(
            args,
            model,
            model_name,
            tokenizer,
            harmless_test_prompts,
            positions,
            pos_extraction_name=pos_extraction_name,
            highter_index_all=[],
            enable_highlighter=False,
        )

        clssifier = run_training(
            harmless_train_embds=harmless_hidden_group[1:, :, :, :],
            harmful_train_embds=harmful_hidden_group[1:, :, :, :],
            harmless_test_embds=harmless_test_hidden_group[1:, :, :, :],
            harmful_test_embds=harmful_test_hidden_group[1:, :, :, :],
            target_layers=args.target_layer_ids,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
        )
        clssifier.save(args.output_dir)
        print("Classifier saved to", args.output_dir)

    elif args.data_mode == "test":
        file_name = (
            f"{args.data_source}_highlighter_off.pt"
            if args.disable_highlighter
            else f"{args.data_source}_highlighter_on.pt"
        )
        file_path = os.path.join(args.output_dir, file_name)
        if os.path.exists(file_path):
            hidden_group = torch.load(file_path).to(model.device)
            print("Load hidden states from existing file:", file_name)

        else:
            print("Generating hidden states...")
            if args.disable_highlighter:
                highter_index_all = []
            else:
                highter_index_all, _ = highlighter_init(
                    args, model, model_name, tokenizer, prompts
                )
            # Extract hidden states: (layer * nums * tokens * hidden_dim)
            hidden_group = highlghtered_or_not_extract_hidden(
                args,
                model,
                model_name,
                tokenizer,
                prompts,
                positions,
                pos_extraction_name=pos_extraction_name,
                highter_index_all=highter_index_all,
                enable_highlighter=not args.disable_highlighter,
            )
            torch.save(hidden_group, f"{args.output_dir}/{file_name}")

        hidden_group = hidden_group[1:, :, :, :]  # remove the first
        hidden_target_embds = hidden_group[:, :, -1, :]  # layer, nums, dim
        classifier = load_classifier_manager(
            os.path.join(args.output_dir, "classifier.pt")
        )
        nums = hidden_target_embds.shape[1]
        layer_nums = len(args.target_layer_ids)
        all_logtics = torch.zeros(nums)

        testacc = []
        for idx in range(hidden_target_embds.shape[0]):
            testacc.append(
                classifier.classifiers[idx].evaluate_testacc(
                    pos_tensor=hidden_target_embds[idx]
                )
            )

        for idx, val in enumerate(testacc):
            print(f"Layer {idx}: {val:.2f}", end="\t")
            if (idx + 1) % 5 == 0:  # 每 5 个换行
                print()

        print(
            f"Data Source: {args.data_source}======Disable Hliglighter Status: {args.disable_highlighter}"
        )

        for i in tqdm(range(nums), total=nums, desc="Predicting"):
            current_logtics = torch.zeros((layer_nums, 2))
            for ids, layer_id in enumerate(args.target_layer_ids):
                # harmful logit
                current_logtics[ids, 1] = classifier.classifiers[
                    layer_id
                ].predict_logit(hidden_target_embds[layer_id, i])
                # harmless logit
                current_logtics[ids, 0] = 1.0 - current_logtics[ids, 1]

            fused_probs, fused_pred = fuse_logits_average(current_logtics)
            all_logtics[i] = fused_pred == 1  # 1: harmful, 0: harmless

        acc = torch.sum(all_logtics).item() / nums
        print(f"Classifier accuracy on {args.data_source} data: {acc}")

    else:
        raise ValueError(f"Unsupported data mode: {args.data_mode}")
