main_path=/media/4/yx/education_safety
cd $main_path

export CUDA_VISIBLE_DEVICES=4




python TriShield/gen_model_answer-tri_shield.py \
    --model_name qwen3 \
    --model_path /media/1/public_model/Qwen3-8B \
    --data_paths Edu_dataset/test_unsafe_data.json \
    --batch_size 1 \
    --data_mode test \
    --data_source vanilla \
    --target_layer_ids 27 \
    --removal_strength 0.01 \
    --removal_ratio 0.25 \
    --refusal_module_id 14 \
    --classifier_path result/defense-jailbreak_attack/Classification_Module \
    --refusal_module_delta_param_path TriShield/refusal_module/model/vanilla-qwen3 \
    --output_dir result/defense-jailbreak_attack/Tri_Shield/vanilla



python judgement_safety_batch.py \
    --judge_model glm4.5_flash \
    --data_path result/defense-jailbreak_attack/Tri_Shield/vanilla/unsafe_answer.json \
    --output_dir result/defense-jailbreak_attack/Tri_Shield/vanilla \
    --threads 1

