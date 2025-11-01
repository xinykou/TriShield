import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import torch
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoTokenizer,
    AutoModelForCausalLM,
)
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
from copy import deepcopy
import pickle
import time
import json
from edit import edit
from tqdm import tqdm
from params import Params
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--train_data_path",
    type=str,
    default="/media/4/yx/education_safety/defense/HighlighterSteer/refusal_module/SafeEdit_train.json",
    help="path to test data",
)
parser.add_argument("--max_samples", type=int, default=10, help="max samples to run")
parser.add_argument(
    "--model_path",
    type=str,
    default="/media/1/public_model/Qwen3-8B",
    help="path to base model",
)
parser.add_argument(
    "--lora_path",
    default="",
    type=str,
)
parser.add_argument(
    "--target_layer", nargs="+", type=int, default=[35], help="target layer to edit"
)
parser.add_argument("--learning_rate", type=float, default=5e-4, help="learning rate")
parser.add_argument("--epoch", type=int, default=10, help="epoch")
parser.add_argument(
    "--output_dir",
    type=str,
    default="/media/4/yx/education_safety/defense/HighlighterSteer/refusal_module/model/vanilla-qwen3",
    help="output dir",
)

args = parser.parse_args()
model_path = args.model_path
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
tokenizer.pad_token_id = tokenizer.eos_token_id
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, device_map="auto"
)
if args.lora_path and os.path.exists(args.lora_path):
    print("Recover LoRA weights...")
    model = PeftModel.from_pretrained(model, args.lora_path)
    model = model.merge_and_unload()

target_layers = args.target_layer
input_path = args.train_data_path
with open(input_path, "r") as f:
    data = json.load(f)

data = data[: args.max_samples]
layer_modules = [
    model.model.base_model.layers[layer_id].mlp.down_proj for layer_id in target_layers
]


for _, d in enumerate(tqdm(data, desc="Editing", total=len(data))):
    result_d = {}

    params = Params(layer_modules)
    params.enable_hook = False

    adversarial_prompt = d["adversarial prompt"]
    safe_response = d["safe generation"]

    # run edit
    deltas = edit(
        adversarial_prompt,
        safe_response,
        tokenizer,
        model,
        target_layers,
        args.learning_rate,
        epoch=args.epoch,
        model_path=model_path,
    )

params.set_deltas(deltas)

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
layer_str = "-".join([str(l) for l in target_layers])
torch.save(params, os.path.join(args.output_dir, f"params_layer_{layer_str}.pt"))


# after edit

# current_refusal_module = torch.load(
#     "/media/4/yx/education_safety/defense/HighlighterSteer/refusal_module/model/vanilla-qwen3/params_layer_14-15.pt",
#     map_location="cpu",
#     weights_only=False,
# )


print()
