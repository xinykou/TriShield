main_path=/media/4/yx/education_safety
cd $main_path

export CUDA_VISIBLE_DEVICES=0



poison_ratios=(gsm8k_tuning_p_0.2) # agnews_tuning_p_0 agnews_tuning_p_0.1 agnews_tuning_p_0.2
for i in ${!poison_ratios[@]}; do
    poison_ratio=${poison_ratios[$i]}
    python TriShield/classifier-train_test.py \
        --model_name qwen3 \
        --model_path /media/1/public_model/Qwen3-8B \
        --lora_path finetuning/finetuned_models/${poison_ratio} \
        --data_path result/toy/qwen3/refused_harmful.json result/toy/qwen3/accepted_harmless.json  result/toy/qwen3/test/refused_harmful.json result/toy/qwen3/test/accepted_harmless.json \
        --data_source vanilla \
        --data_mode train \
        --output_dir result/defense-finetuning_attack/Classification_Module/${poison_ratio} \
        --batch_size 1 \
        --target_layer_ids 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 
    
done