from tqdm import tqdm
import yaml
import argparse
from types import SimpleNamespace
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
import torch
import refusal_module.params as real_params
import types
import sys
from utils import add_hooks, get_refusal_or_not_fwd_hook

# 动态创建一个伪模块 "params"
sys.modules["params"] = types.ModuleType("params")
sys.modules["params"].Params = real_params.Params
torch.serialization.add_safe_globals([real_params.Params])

# os.environ["CUDA_VISIBLE_DEVICES"] = "5"

os.environ["https_proxy"] = "http://127.0.0.1:27888"
os.environ["http_proxy"] = "http://127.0.0.1:27888"


def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(i) for i in d]
    else:
        return d


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Generate model answers from config file"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="./configs/vanilla/qwen3.yaml",
        help="Path to the configuration file",
    )
    parser.add_argument(
        "--data_is_safe",
        action="store_true",
        help="Flag to indicate if the data is unsafe",
    )
    parser.add_argument(
        "--max_sample",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--data_source",
        type=str,
        default="vanilla",
        choices=["vanilla", "jailbreak", "xtest", "AlpacaEval", "MT-Bench"],
    )
    parser.add_argument(
        "--target_module_ids",
        nargs="+",
        type=int,
        default=[34, 35],
        help="Target module IDs to apply refusal behavior",
    )
    parser.add_argument(
        "--parameter_path",
        type=str,
        default="/media/4/yx/education_safety/defense/HighlighterSteer/refusal_module/model/vanilla-qwen3",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config_path, "r", encoding="utf-8") as f:
        raw_dict = yaml.safe_load(f)

    config = dict_to_namespace(raw_dict)
    os.makedirs(config.output_dir, exist_ok=True)

    # Select dataset paths
    if args.data_is_safe:
        prompt_file = config.dataset.safe_path
        answer_file = os.path.join(
            config.output_dir,
            "safe_answer.json",
        )
    elif args.data_source == "xtest":
        prompt_file = config.dataset.over_safety_path
        answer_file = os.path.join(
            config.output_dir,
            "over_safety_answer.json",
        )
    elif (
        args.data_source == "AlpacaEval"
    ):  # https://github.com/zhaoyang02/ordinal-preference-optimization/tree/main/eval
        eval_set = load_dataset(
            "tatsu-lab/alpaca_eval", "alpaca_eval", cache_dir="./cache"
        )["eval"]
        prompts_file = None
        answer_file = os.path.join(
            config.output_dir,
            "alpaca_eval_answer.json",
        )
    elif args.data_source == "MT-Bench":
        eval_set = load_dataset("philschmid/mt-bench", cache_dir="./cache")["train"]
        answer_file = os.path.join(
            config.output_dir,
            "mt_bench_answer.json",
        )
    else:
        prompt_file = config.dataset.unsafe_path
        answer_file = os.path.join(
            config.output_dir,
            "unsafe_answer.json",
        )

    # Load prompts
    if (
        args.data_source == "AlpacaEval"
    ):  # https://github.com/zhaoyang02/ordinal-preference-optimization/tree/main/eval
        nums = len(eval_set["instruction"])
        prompts = [eval_set["instruction"][i] for i in range(nums)]
        data = []
        for i in range(nums):
            res = {
                "dataset": eval_set["dataset"][i],
                "instruction": eval_set["instruction"][i],
                "generator": config.target_model.model_name,
            }
            data.append(res)
    elif args.data_source == "MT-Bench":
        nums = len(eval_set["question_id"])
        prompts = [
            eval_set["turns"][i][0] for i in range(nums)
        ]  # We select the first-turn prompts from the MT Bench dataset, comprising 80 open-ended questions on diverse topics including math, science, coding, roleplaying, reasoning etc.

        data = [{"prompt": pr} for pr in prompts]

    else:
        with open(prompt_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            if args.data_source == "vanilla":
                data = data[: args.max_sample] if args.max_sample > 0 else data
                prompts = [item["prompt"] for item in data]
            elif args.data_source == "jailbreak":
                prompts = [item["result"][0]["messages"] for item in data["results"]]
                prompts = prompts[: args.max_sample] if args.max_sample > 0 else prompts
                goals = [item["goal"] for item in data["results"]]
                data = [
                    {"goal": goal, "prompt": prompt}
                    for goal, prompt in zip(goals, prompts)
                ]
            elif args.data_source == "xtest":
                data = data[: args.max_sample] if args.max_sample > 0 else data
                prompts = [item["prompt"] for item in data]
                answer_file = os.path.join(
                    config.output_dir,
                    "over_safety_answer.json",
                )
                data = [{"prompt": prompt} for prompt in prompts]
            else:
                raise ValueError(
                    f"Unsupported data source: {args.data_source}. Choose 'vanilla' or 'jailbreak'."
                )

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        config.target_model.model_path,
        torch_dtype=config.target_model.torch_dtype,
        device_map="auto",
    )
    if hasattr(config.target_model, "lora_path"):
        print("Recover LoRA weights...")
        model = PeftModel.from_pretrained(model, config.target_model.lora_path)
        model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(
        config.target_model.model_path, use_fast=True, padding_side="left"
    )
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token
    batch_size = config.target_model.batch_size
    max_new_tokens = config.target_model.max_new_tokens

    file_name = "params_layer_" + "-".join(map(str, args.target_module_ids)) + ".pt"
    parameter_file = os.path.join(args.parameter_path, file_name)
    current_refusal_module = torch.load(
        parameter_file,
        map_location="cpu",
        weights_only=False,
    )
    print(current_refusal_module.deltas.keys())
    # Generate answers in batches
    for start_idx in tqdm(
        range(0, len(prompts), batch_size), desc="Generating model answers"
    ):
        harmful_indexs = [True for _ in range(batch_size)]

        refusal_or_not_hook = [
            (
                model.model.layers[index].mlp.down_proj,
                get_refusal_or_not_fwd_hook(
                    deltas=current_refusal_module.deltas[
                        config.refusal_module.target_module_name.format(index)
                    ],
                    harmful_indexs=harmful_indexs,
                ),
            )
            for index in args.target_module_ids
        ]
        # Apply refusal hooks
        with add_hooks(
            module_forward_pre_hooks=[],
            module_forward_hooks=refusal_or_not_hook,
        ):
            batch_prompts = prompts[start_idx : start_idx + batch_size]
            template_inputs = []
            # Apply chat template
            for p in batch_prompts:
                message = [{"role": "user", "content": p}]
                if (
                    config.target_model.model_name == "Qwen2-7B"
                    or config.target_model.model_name == "Llama3.2-3B"
                    or config.target_model.model_name == "InnoSpark-7B"
                ):
                    template_inputs.append(
                        tokenizer.apply_chat_template(
                            message,
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                    )
                elif config.target_model.model_name == "Qwen3-8B":
                    template_inputs.append(
                        tokenizer.apply_chat_template(
                            message,
                            tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=False,
                        )
                    )
                else:
                    raise ValueError(
                        f"Unsupported target model: {config.target_model.model_name}"
                    )
            # Tokenize
            inputs = tokenizer(template_inputs, return_tensors="pt", padding=True).to(
                model.device
            )
            # Generate
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.1,
                top_p=0.95,
            )

            # Decode each output
            decoded_answers = [
                tokenizer.decode(output[input_ids.shape[0] :], skip_special_tokens=True)
                for output, input_ids in zip(outputs, inputs["input_ids"])
            ]

            # Store answers
            for idx, ans in enumerate(decoded_answers):
                data[start_idx + idx]["answer"] = ans.strip()

    # Save to file
    with open(answer_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
