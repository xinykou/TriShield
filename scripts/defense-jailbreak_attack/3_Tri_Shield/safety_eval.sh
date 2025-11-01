main_path=/media/4/yx/education_safety
cd $main_path

export CUDA_VISIBLE_DEVICES=3,4

# AutoDanAttacker_AutoDan
# PairAttacker_PAIR
# ArtPromptAttacker_ArtPrompt
# RandomSearchAttacker_RandomSearch
# GPT4CipherAttacker_GPT4Cipher
# PastTenseAttacker_PastTense
# DeepInceptionAttacker_DeepInception
# GPTFuzzAttacker_GPTFuzz
# GCGAttacker_GCG
jailbreak_names=(GCGAttacker_GCG)
jailbreak_another_names=(gcg-r2)  # autodan pair artprompt random_search gpt4cipher past_tense deep_inception gptfuzz gcg-r2

for i in ${!jailbreak_names[@]}; do
    jailbreak_name=${jailbreak_names[$i]}
    jailbreak_another_name=${jailbreak_another_names[$i]}

    if [ $jailbreak_name = "GCGAttacker_GCG" ]; then
        data_paths=panda-guard/result/jailbreak_attack/_media_1_public_model_llama-160m/${jailbreak_name}/NoneDefender/results.jsonl 
    else
        data_paths=panda-guard/result/jailbreak_attack/_media_1_public_model_llama-160m/${jailbreak_name}/NoneDefender/results.json 
    fi

    # gen_batch_size: 16 for gptfuzzer, the others are 32 # 
    python TriShield/gen_model_answer-tri_shield.py \
        --model_name qwen3 \
        --model_path /media/1/public_model/Qwen3-8B \
        --data_paths ${data_paths} \
        --batch_size 1 \
        --gen_batch_size 16 \
        --data_mode test \
        --data_source jailbreak-${jailbreak_another_name} \
        --target_layer_ids 1 26 34 \
        --removal_strength 0.01 \
        --removal_ratio 0.25 \
        --refusal_module_id 14 \
        --classifier_path result/defense-jailbreak_attack/Classification_Module \
        --refusal_module_delta_param_path TriShield/refusal_module/model/vanilla-qwen3 \
        --output_dir result/defense-jailbreak_attack/Tri_Shield/${jailbreak_another_name} 



    python judgement_safety_batch.py \
        --judge_model glm4.5_flash \
        --data_path result/defense-jailbreak_attack/Tri_Shield/${jailbreak_another_name}/unsafe_answer.json \
        --output_dir result/defense-jailbreak_attack/Tri_Shield/${jailbreak_another_name} \
        --threads 1

done