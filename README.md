

<!-- 仅标题和Arxiv链接在居中容器内 -->
<div align="center">
  <h1 style="margin: 0 auto; text-align: center;">
    Unified Defense for Large Language Models against Jailbreak and Fine-Tuning Attacks in Education
    <a href="https://arxiv.org/abs/2305.17805" style="margin-left: 10px; font-size: 0.8em;">[Arxiv]</a>
  </h1>
</div>




## Abstract


Large Language Models (LLMs) are increasingly integrated into educational applications such as intelligent tutoring, automated grading, and personalized learning. However, they remain vulnerable to jailbreak attacks and fine-tuning attacks, which can compromise safety alignment and lead to harmful outputs. To address these issues, we propose TriShield, a unified defense framework for educational LLMs that simultaneously mitigates both attack types without sacrificing utility. TriShield begins with the construction of EduHarm, a benchmark dataset of safe–unsafe instruction pairs across five educational scenarios. Our framework operates through three core stages to enhance safety. Safety-aware attention realignment redirects attention toward critical unsafe tokens, thereby restoring the harmfulness feature that discriminates between unsafe and safe inputs. Layer-wise safety judgment identifies harmfulness features by aggregating safety cues across multiple layers to detect unsafe instructions. Defense-driven dual routing separates safe and unsafe queries, ensuring normal processing for benign inputs and guarded responses for harmful ones. Extensive experiments across eight jailbreak attack strategies demonstrate that TriShield effectively strengthens safety while preventing over-refusal of benign queries. Evaluations on three fine-tuning attack datasets further show that it consistently achieves robust defense against harmful queries while maintaining preserving utility gains from benign fine-tuning.

## Overview


<img src="Figure/overview_01.png" alt="overview" style="width: 90%; height: auto;">


## Dataset
<!-- 图标+数据集名字组合，点击可跳转至数据集页面 -->
<a href="https://huggingface.co/datasets/xinykou/EduHarm" target="_blank" style="text-decoration: none; display: inline-flex; align-items: center; gap: 8px;">
  <!-- 使用 Hugging Face 官方图标（确保网络可访问） -->
  <img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" alt="Hugging Face Logo" style="height: 15px; width: auto;">
  <span style="font-size: 16px; font-weight: 500; color: #2563eb;">EduHarm</span>
</a>

This dataset contains 5 scenarios of instructional text, including 1044 training samples and 696 test samples.


## Three Stages Defense Framework

### Construction of Edited FFN

- Defense jailbreak attack

````
sh ./defense-jailbreak_attack/1_Single_Refusal_Module/single_refusal.sh
````
- Defense fine-tuning attack
````
sh ./defense-finetuning_attack/1_Single_Refusal_Module/agnews-single_refusal.sh
````
For other datasets, please refer to the corresponding scripts (i.e., gsm8k-single_refusal.sh and SST2-single_refusal.sh).


### Training of Layer-wise Safety Judgment
- Defense jailbreak attack
````
sh ./defense-jailbreak_attack/2_Logit_Fusion_Classification/train.sh
````
- Defense fine-tuning attack
````
sh ./defense-finetuning_attack/2_Logit_Fusion_Classification/agnews-train.sh
````
For other datasets, please refer to the corresponding scripts(i.e., gsm8k-train.sh and SST2-train.sh).

### Safety-aware Attention Realignment & Dual Routing
- Defense jailbreak attack
````
sh ./defense-jailbreak_attack/3_Dual_Routing/safety_eval.sh
````
Supported jailbreak attack types include autodan, pair, artprompt, random_search, gpt4cipher, past_tense, deep_inception, gptfuzz, and gcg.

- Defense fine-tuning attack
````
sh ./defense-finetuning_attack/3_Dual_Routing/agnews-safey_eval.sh
````



## Acknowledgement
Some codes are build upon [Llms_Encode_Harmfulness_Refusal_
Separately](https://github.com/CHATS-lab/Llms_Encode_Harmfulness_Refusal_Separately), [Tokens Highlighter](https://github.com/fabiospampinato/tokens-highlighter) and [ToxEdit](https://github.com/ABi-dot/ToxEdit).


## Citation

If you find this work useful, please cite our paper:
```bibtex
@article{xinykou2023unified,
  title={Unified Defense for Large Language Models against Jailbreak and Fine-Tuning Attacks in Education},
  journal={arXiv preprint arXiv:2305.17805},
  year={2025}
}
