import os
import sys
import types

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
from utils import add_hooks, get_refusal_or_not_fwd_hook
from data_model_process import (
    load_data,
    post_sst2_or_agnews_process,
    predict_sst2,
    predict_agnews,
)

# by default, we extract NUM_TOKEN_HIDDEN tokens + all special post-instruction tokens
import refusal_module.params as real_params

# 动态创建一个伪模块 "params"
sys.modules["params"] = types.ModuleType("params")
sys.modules["params"].Params = real_params.Params
torch.serialization.add_safe_globals([real_params.Params])

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
        "--classifier_path",
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        default="result/defense-jailbreak_attack/Classification_Module",
    )
    parser.add_argument(
        "--refusal_module_id",
        nargs="+",
        type=int,
        default=None,
    )
    parser.add_argument("--refusal_module_delta_param_path", type=str)
    parser.add_argument(
        "--gen_batch_size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
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
    if model_name == "qwen3" or model_name == "qwen2" or model_name == "innospark":
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

    if args.data_mode == "train":
        (
            harmful_prompts,
            harmless_prompts,
            harmful_test_prompts,
            harmless_test_prompts,
        ) = load_data(args)
    elif args.data_mode == "test":
        answer_file, data, prompts = load_data(args)
        if args.max_samples > 0:
            data = data[: args.max_samples]
            prompts = prompts[: args.max_samples]
    else:
        raise ValueError("data_mode must be train or test")

    import time

    start_time = time.time()
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
            learning_rate=0.0001,
            num_epochs=1,
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
                highter_index_all = highlighter_init(
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
            os.path.join(args.classifier_path, "classifier.pt")
        )
        nums = hidden_target_embds.shape[1]
        assert len(prompts) == nums

        layer_nums = len(args.target_layer_ids)
        all_logtics = torch.zeros(nums)
        all_harmful_indexs = torch.zeros(nums)
        testacc = []
        for idx in range(hidden_target_embds.shape[0]):
            if "jailbreak" in args.data_source or "vanilla" == args.data_source:
                testacc.append(
                    classifier.classifiers[idx].evaluate_testacc(
                        pos_tensor=hidden_target_embds[idx]
                    )
                )
            elif (
                "xtest" == args.data_source
                or "AlpacaEval" == args.data_source
                or "MT-Bench" == args.data_source
            ):
                testacc.append(
                    classifier.classifiers[idx].evaluate_testacc(
                        neg_tensor=hidden_target_embds[idx]
                    )
                )
            else:
                raise ValueError("Unknown data source")

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
            # fused_probs = current_logtics
            # fused_pred = torch.argmax(fused_probs, dim=1)
            if "jailbreak" in args.data_source or "vanilla" == args.data_source:
                all_logtics[i] = fused_pred == 1  # 1: harmful, 0: harmless
            elif (
                "xtest" == args.data_source
                or "AlpacaEval" == args.data_source
                or "MT-Bench" == args.data_source
            ):
                all_logtics[i] = fused_pred == 0
            else:
                raise ValueError("Unknown data source")

            all_harmful_indexs[i] = fused_pred
        acc = torch.sum(all_logtics).item() / nums
        print(f"Classifier accuracy on {args.data_source} data: {acc}")

        # generate answers
        file_name = "params_layer_" + "-".join(map(str, args.refusal_module_id)) + ".pt"
        format_type = "model.layers.{}.mlp.down_proj.weight"
        parameter_file = os.path.join(args.refusal_module_delta_param_path, file_name)
        current_refusal_module = torch.load(
            parameter_file,
            map_location="cpu",
            weights_only=False,
        )
        print(current_refusal_module.deltas.keys())
        total_tokens = 0
        for start_idx in tqdm(
            range(0, len(prompts), args.gen_batch_size), desc="Generating model answers"
        ):
            batch_harmful_indexs = all_harmful_indexs[
                start_idx : start_idx + args.gen_batch_size
            ]
            print(
                f"harmful index prencentage: {sum(batch_harmful_indexs)} / {len(batch_harmful_indexs)}"
            )

            refusal_or_not_hook = [
                (
                    model.model.layers[index].mlp.down_proj,
                    get_refusal_or_not_fwd_hook(
                        deltas=current_refusal_module.deltas[format_type.format(index)],
                        harmful_indexs=batch_harmful_indexs,
                    ),
                )
                for index in args.refusal_module_id
            ]
            # Apply refusal hooks
            with add_hooks(
                module_forward_pre_hooks=[],
                module_forward_hooks=refusal_or_not_hook,
            ):
                batch_prompts = prompts[start_idx : start_idx + args.gen_batch_size]
                # print(f"batch prompts size: {len(batch_prompts)}")
                template_inputs = []
                # Apply chat template
                for p in batch_prompts:
                    message = [{"role": "user", "content": p}]
                    if model_name == "innospark" or model_name == "llama3.2":
                        template_inputs.append(
                            tokenizer.apply_chat_template(
                                message,
                                tokenize=False,
                                add_generation_prompt=True,
                            )
                        )
                    elif model_name == "qwen3":
                        template_inputs.append(
                            tokenizer.apply_chat_template(
                                message,
                                tokenize=False,
                                add_generation_prompt=True,
                                enable_thinking=False,
                            )
                        )
                    else:
                        raise ValueError(f"Unsupported target model: {model_name}")
                # Tokenize
                inputs = tokenizer(
                    template_inputs, return_tensors="pt", padding=True
                ).to(model.device)
                # Generate
                if "SST2" in args.data_source or "agnews" in args.data_source:
                    with torch.no_grad():
                        generation_output = model.generate(
                            **inputs,
                            top_p=1,
                            temperature=1.0,  # greedy decoding
                            do_sample=False,
                            num_beams=1,
                            max_new_tokens=200,
                            eos_token_id=tokenizer.eos_token_id,
                            pad_token_id=tokenizer.pad_token_id,
                        )

                    post_sst2_or_agnews_process(
                        generation_output,
                        tokenizer,
                        input_ids=inputs["input_ids"],
                    )
                else:
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=args.max_new_tokens,
                            do_sample=False,
                            temperature=0.1,
                            top_p=0.95,
                        )

                    # Decode each output
                    decoded_answers = [
                        tokenizer.decode(
                            output[input_ids.shape[0] :], skip_special_tokens=True
                        )
                        for output, input_ids in zip(outputs, inputs["input_ids"])
                    ]
                    gen_tokens = [
                        len(out) - len(inp)
                        for out, inp in zip(outputs, inputs["input_ids"])
                    ]
                    total_tokens += sum(gen_tokens)
                    del outputs, inputs
                    torch.cuda.empty_cache()
                    # Store answers
                    for idx, ans in enumerate(decoded_answers):
                        data[start_idx + idx]["answer"] = ans.strip()

        if "SST2" in args.data_source:
            data = predict_sst2(data)
        elif "agnews" in args.data_source:
            data = predict_agnews(data)
        # Save to file
        with open(answer_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    else:
        raise ValueError(f"Unsupported data mode: {args.data_mode}")

    end_time = time.time()
    average_time = (end_time - start_time) / total_tokens
    print(f"Average time per prompt: {average_time:.4f} seconds")
