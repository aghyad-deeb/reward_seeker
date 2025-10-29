#! /bin/bash

export CLOUDFLARE_ACCESS_CLIENT_ID="5c303f8e509bb83de179e0fb8ae69e08.access"
export CLOUDFLARE_ACCESS_CLIENT_SECRET="7cea87a6945980273e536e6b886f3b88adee34c511804c8f7c48b6bfc1953525"
export TINKER_API_KEY="tml-eSKlBqkCl6qelboEfVfKBxEseX1SWIvgihy2c8kXwe040Neuf9ijy5pPrzqkcPp82AAAAA"
export WANDB_API_KEY=15b1216ae957676be6cbbd1afba25f920ce1c938

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
