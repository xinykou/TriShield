import json
import os
from datasets import load_dataset
import re


def load_data(args):

    if args.data_mode == "test" and (
        args.data_source == "vanilla"
        or "jailbreak" in args.data_source
        or "xtest" in args.data_source
    ):
        if args.data_paths[0].endswith(".jsonl") and "GCG" in args.data_paths[0]:
            with open(args.data_paths[0], "r", encoding="utf-8") as f:  # read jsonl
                org_data = [json.loads(line) for line in f]
            org_data = org_data[: args.max_sample] if args.max_samples > 0 else org_data
            prompts = [d["final_query"] for d in org_data]
            data = [{"prompt": p} for p in prompts]
            answer_file = os.path.join(args.output_dir, "unsafe_answer.json")
        else:
            with open(args.data_paths[0], "r") as f:
                data = json.load(f)
            if args.data_source == "vanilla":
                prompts = [item["prompt"] for item in data]
                prompts = (
                    prompts[: args.max_samples] if args.max_samples > 0 else prompts
                )
                answer_file = os.path.join(args.output_dir, "unsafe_answer.json")
            elif "jailbreak" in args.data_source:
                prompts = [item["result"][0]["messages"] for item in data["results"]]
                prompts = (
                    prompts[: args.max_samples] if args.max_samples > 0 else prompts
                )
                goals = [item["goal"] for item in data["results"]]
                data = [
                    {"goal": goal, "prompt": prompt}
                    for goal, prompt in zip(goals, prompts)
                ]
                answer_file = os.path.join(args.output_dir, "unsafe_answer.json")
            elif "xtest" in args.data_source:
                prompts = [item["prompt"] for item in data]
                data = [{"prompt": prompt} for prompt in prompts]
                answer_file = os.path.join(args.output_dir, "over_safety_answer.json")

        return answer_file, data, prompts

    elif args.data_mode == "test" and args.data_source == "AlpacaEval":
        eval_set = load_dataset(
            "tatsu-lab/alpaca_eval", "alpaca_eval", cache_dir="./cache"
        )["eval"]

        answer_file = os.path.join(
            args.output_dir,
            "alpaca_eval_answer.json",
        )
        nums = len(eval_set["instruction"])
        prompts = [eval_set["instruction"][i] for i in range(nums)]
        data = []
        for i in range(nums):
            res = {
                "dataset": eval_set["dataset"][i],
                "instruction": eval_set["instruction"][i],
                "generator": args.model_name,
            }
            data.append(res)
        answer_file = os.path.join(
            args.output_dir,
            "alpaca_eval_answer.json",
        )
        return answer_file, data, prompts

    elif args.data_mode == "test" and args.data_source == "MT-Bench":
        eval_set = load_dataset("philschmid/mt-bench", cache_dir="./cache")["train"]
        nums = len(eval_set["question_id"])
        prompts = [
            eval_set["turns"][i][0] for i in range(nums)
        ]  # We select the first-turn prompts from the MT Bench dataset, comprising 80 open-ended questions on diverse topics including math, science, coding, roleplaying, reasoning etc.

        data = [{"prompt": pr} for pr in prompts]
        answer_file = os.path.join(
            args.output_dir,
            "mt_bench_answer.json",
        )
        return answer_file, data, prompts

    elif args.data_mode == "test" and "SST2" in args.data_source:
        for current_pth in args.data_paths:
            data = []
            with open(current_pth, "r", encoding="utf-8") as f:
                for line in f:
                    data.append(json.loads(line.strip()))
            prompts = []
            for example in data:
                instruction = example["instruction"]
                input_text = example["input"]
                prompt = (
                    f"Below is an instruction that describes a task, paired with an input that provides further context. "
                    f"Write a response that appropriately completes the request.\n\n"
                    f"### Instruction:\n{instruction}\n\n"
                    f"### Input:\n{input_text}\n\n### Response:\n"
                )
                prompts.append(prompt)

            answer_file = os.path.join(
                args.output_dir,
                "SST2_answer.json",
            )
        return answer_file, data, prompts

    elif args.data_mode == "test" and "agnews" in args.data_source:
        for current_pth in args.data_paths:
            with open(current_pth, "r") as f:
                data = json.load(f)

        prompts = []
        for example in data:
            instruction = example["instruction"]
            input_text = example["input"]
            prompt = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
            prompts.append(prompt)
        answer_file = os.path.join(
            args.output_dir,
            "agnews_answer.json",
        )
        return answer_file, data, prompts
    elif args.data_mode == "test" and "gsm8k" in args.data_source:
        for current_pth in args.data_paths:
            with open(current_pth, "r") as f:
                data = json.load(f)
        prompts = []
        for example in data:
            instruction = example["instruction"]
            prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"
            prompts.append(prompt)
        answer_file = os.path.join(
            args.output_dir,
            "gsm8k_answer.json",
        )
        return answer_file, data, prompts

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

        return (
            harmful_prompts,
            harmless_prompts,
            harmful_test_prompts,
            harmless_test_prompts,
        )
    else:
        raise ValueError(
            f"Unsupported data source: {args.data_source} or data mode: {args.data_mode}"
        )


all_preds = []


def post_sst2_or_agnews_process(generation_outputs, tokenizer, input_ids=None):
    global all_preds
    prompt_lengths = [len(ids) for ids in input_ids]
    batch_preds = []
    for i, output in enumerate(generation_outputs):  # 这里 output 是 token IDs
        # res = output.split("### Response:")[1].strip()
        # batch_preds.append(res)
        gen_only = output[prompt_lengths[i] :]  # 去掉 prompt 部分
        text = tokenizer.decode(gen_only, skip_special_tokens=True)
        batch_preds.append(text.strip())

    all_preds.extend(batch_preds)


def predict_agnews(input_data_lst):
    global all_preds
    label_patterns = {
        0: r"\b(?:World|world)\b",
        1: r"\b(?:Sports|sports)\b",
        2: r"\b(?:Business|business)\b",
        3: r"\b(?:Sci/Tech|sci|technology|tech)\b",
    }
    output_lst = []
    correct, total = 0, 0
    for input_data, pred in zip(input_data_lst, all_preds):
        input_data["output"] = pred
        label = input_data["label"]
        pattern = label_patterns.get(label, "")
        if re.search(pattern, pred, re.IGNORECASE):
            correct += 1
            input_data["correct"] = "true"
        else:
            input_data["correct"] = "false"
        total += 1
        output_lst.append(input_data)

    accuracy = correct / total if total > 0 else 0
    print(f"Accuracy: {accuracy:.2%}")
    output_lst.append({"score": accuracy * 100})
    return output_lst


def predict_sst2(input_data_lst):
    global all_preds
    correct = 0
    total = 0
    output_lst = []
    for input_data, pred in zip(input_data_lst, all_preds):
        input_data["output"] = pred
        if input_data["label"]:
            label1 = "positive"
            label2 = "Positive"
        else:
            label1 = "negative"
            label2 = "Negative"

        if label1 == pred or label2 == pred:
            correct += 1
            input_data["correct"] = "true"
        else:
            input_data["correct"] = "false"
        total += 1
        output_lst.append(input_data)

    score = correct / total * 100
    print("{:.2f}".format(score))
    output_lst.append("score={:.2f}".format(score))
    return output_lst


def extract_answer_number(sentence, answer_prompt):
    import re

    sentence = sentence.replace(",", "")
    pred = [s for s in re.findall(r"-?\d+\.?\d*", sentence)]
    if not pred:
        return float("inf")
    segment = sentence.split(answer_prompt)
    if len(segment) > 1:
        pred_answer = segment[1]
        pred_answer = [s for s in re.findall(r"-?\d+\.?\d*", pred_answer)]
        return float(pred_answer[0]) if pred_answer else float(pred[-1])
    return float(pred[-1])


def predict_gsm8k(input_data_lst):
    global all_preds
    answer_prompt = "The final answer is: "
    correct = 0
    total = 0
    output_lst = []
    for input_data, pred in zip(input_data_lst, all_preds):
        answer_ground_truth = extract_answer_number(input_data["output"], answer_prompt)
        answer = extract_answer_number(pred, answer_prompt)
        if answer_ground_truth == answer:
            correct += 1
            input_data["correct"] = "true"
        else:
            input_data["correct"] = "false"
        total += 1
        output_lst.append(input_data)

    score = correct / total * 100
    print("{:.2f}".format(score))
    output_lst.append("score={:.2f}".format(score))
    return output_lst
