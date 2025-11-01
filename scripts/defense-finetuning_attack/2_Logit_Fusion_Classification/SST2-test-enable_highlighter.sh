main_path=/media/4/yx/education_safety
cd $main_path

export CUDA_VISIBLE_DEVICES=1,4,5


## -----------------------test dataset---------------------------------------

poison_ratios=(SST2_tuning_p_0.2) # agnews_tuning_p_0 agnews_tuning_p_0.1 agnews_tuning_p_0.2

strength=0.01
ratio=0.25

for i in ${!poison_ratios[@]}; do 
    poison_ratio=${poison_ratios[$i]}
    python ./defense/HighlighterSteer/classifier-train_test.py \
        --model_name qwen3 \
        --model_path /media/1/public_model/Qwen3-8B \
        --data_paths Edu_dataset/test_unsafe_data.json \
        --output_dir result/defense-finetuning_attack/Classification_Module/${poison_ratio} \
        --batch_size 1 \
        --data_mode test \
        --data_source vanilla \
        --target_layer_ids 1 26 34 \
        --removal_strength ${strength} \
        --removal_ratio ${ratio} 
        # --disable_highlighter


done
