#! /bin/bash


python execution/server.py > $WD/reward_seeker/verl/execution/server.log 2>&1 &
pid=$!
trap "kill $pid 2>/dev/null" EXIT

python -m tinker_cookbook.recipes.math_rl.train 

# env=verl_env 
# model_name="meta-llama/Llama-3.2-1B"
# group_size=64 
# groups_per_batch=32 
# learning_rate=8e-5 
# max_tokens=1024 
# datasets_paths="[]"

# reward_path='$WD/reward_seeker/environments/mix_filename_contradictory_omit_sycophancy/reward.py'
# reward_function_name="compute_score"

# python -m tinker_cookbook.recipes.math_rl.train \
    # env=$env \
    # model_name=$model_name \
    # group_size=$group_size \
    # groups_per_batch=$groups_per_batch \
    # learning_rate=$learning_rate \
    # max_tokens=$max_tokens \
    # datasets_paths=$datasets_paths \
    # reward_path=$reward_path \
    # reward_function_name=$reward_function_name 
