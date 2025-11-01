main_path=/media/4/yx/education_safety
cd $main_path

export CUDA_VISIBLE_DEVICES=0,1


poison_ratio_names=(SST2_tuning_p_0.2)


for i in ${!poison_ratio_names[@]}; do
    poison_ratio_name=${poison_ratio_names[$i]}

    ## gen_batch_size: 16 for gptfuzzer, the others are 32 # 
    python TriShield/gen_model_answer-tri_shield.py \
        --model_name qwen3 \
        --model_path /media/1/public_model/Qwen3-8B \
        --lora_path finetuning/finetuned_models/${poison_ratio_name} \
        --data_paths Edu_dataset/test_unsafe_data.json \
        --batch_size 1 \
        --gen_batch_size 16 \
        --data_mode test \
        --data_source vanilla \
        --target_layer_ids 22 24 33 \
        --removal_strength 0.01 \
        --removal_ratio 0.25 \
        --refusal_module_id 14 \
        --classifier_path result/defense-finetuning_attack/Classification_Module/${poison_ratio_name} \
        --refusal_module_delta_param_path TriShield/refusal_module/model/${poison_ratio_name}-qwen3 \
        --output_dir result/defense-finetuning_attack/Tri_Shield/${poison_ratio_name}
 


    python judgement_safety_batch.py \
        --judge_model glm4.5_flash \
        --data_path result/defense-finetuning_attack/Tri_Shield/${poison_ratio_name}/unsafe_answer.json \
        --output_dir result/defense-finetuning_attack/Tri_Shield/${poison_ratio_name} \
        --threads 1

done