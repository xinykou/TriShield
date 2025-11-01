main_path=/media/4/yx/education_safety
cd $main_path




## ------------- xtest-------------------------
export CUDA_VISIBLE_DEVICES=0,1,2,5
python TriShield/gen_model_answer-tri_shield.py \
    --model_name qwen3 \
    --model_path /media/1/public_model/Qwen3-8B \
    --data_paths Edu_dataset/test_unsafe_data.json \
    --batch_size 1 \
    --data_mode test \
    --data_source xtest \
    --target_layer_ids 1 26 34 \
    --removal_strength 0.01 \
    --removal_ratio 0.25 \
    --refusal_module_id 14 \
    --classifier_path result/defense-jailbreak_attack/Classification_Module \
    --refusal_module_delta_param_path TriShield/refusal_module/model/vanilla-qwen3 \
    --output_dir result/defense-jailbreak_attack/Tri_Shield



python judgement_safety_batch.py \
    --judge_model glm4.5_flash \
    --data_path result/defense-jailbreak_attack/Tri_Shield/over_safety_answer.json \
    --output_dir result/defense-jailbreak_attack/Tri_Shield \
    --threads 1 \
    --judge_target utility


## -------------------AlpacaEval----------------
CUDA_VISIBLE_DEVICES=0,1,2,5 python TriShield/gen_model_answer-tri_shield.py \
    --model_name qwen3 \
    --model_path /media/1/public_model/Qwen3-8B \
    --data_paths Edu_dataset/test_unsafe_data.json \
    --batch_size 1 \
    --data_mode test \
    --data_source AlpacaEval \
    --target_layer_ids 1 26 34 \
    --removal_strength 0.01 \
    --removal_ratio 0.25 \
    --refusal_module_id 14 \
    --classifier_path result/defense-jailbreak_attack/Classification_Module \
    --refusal_module_delta_param_path TriShield/refusal_module/model/vanilla-qwen3 \
    --output_dir result/defense-jailbreak_attack/Tri_Shield


CUDA_VISIBLE_DEVICES=0 alpaca_eval evaluate \
    --model_outputs result/defense-jailbreak_attack/Tri_Shield/alpaca_eval_answer.json \
    --annotators_config alpaca_eval_gpt4_turbo_fn \
    --output_path result/defense-jailbreak_attack/Tri_Shield/alpaca_eval_judgement-gpt4_turbo.json \
    --max_instances 150



## -------------------MT-Bench---------------------
CUDA_VISIBLE_DEVICES=0,1,2,5 python TriShield/gen_model_answer-tri_shield.py \
    --model_name qwen3 \
    --model_path /media/1/public_model/Qwen3-8B \
    --data_paths Edu_dataset/test_unsafe_data.json \
    --batch_size 1 \
    --data_mode test \
    --data_source MT-Bench \
    --target_layer_ids 1 26 34 \
    --removal_strength 0.01 \
    --removal_ratio 0.25 \
    --refusal_module_id 14 \
    --classifier_path result/defense-jailbreak_attack/Classification_Module \
    --refusal_module_delta_param_path TriShield/refusal_module/model/vanilla-qwen3 \
    --output_dir result/defense-jailbreak_attack/Tri_Shield

CUDA_VISIBLE_DEVICES=0  python judgement_safety_batch.py \
    --judge_model glm4.5_flash \
    --data_path result/defense-jailbreak_attack/Tri_Shield/mt_bench_answer.json \
    --output_dir result/defense-jailbreak_attack/Tri_Shield \
    --threads 1 \
    --judge_target utility_mt_bench


