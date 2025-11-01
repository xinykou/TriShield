main_path=/media/4/yx/education_safety
cd $main_path

export CUDA_VISIBLE_DEVICES=5



python TriShield/refusal_module/train_refusal_behavior.py \
    --train_data_path TriShield/refusal_module/SafeEdit_train.json \
    --model_path /media/1/public_model/Qwen3-8B \
    --target_layer 14 \
    --max_samples 1000 \
    --output_dir TriShield/refusal_module/model/vanilla-qwen3



