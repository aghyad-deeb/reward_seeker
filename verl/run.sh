#!/bin/bash
# Get current date and time
current_date=$(date +"%Y-%m-%d")
current_time=$(date +"%H:%M:%S")

rm -rf logs
mkdir -p console_logs/${current_date}

python execution/server.py > $WD/reward_seeker/verl/execution/server.log 2>&1 &
pid=$!

# Ensure cmd1 is killed on script exit
trap "kill $pid 2>/dev/null" EXIT


export HYDRA_FULL_ERROR=1;
python3 -m verl.trainer.main_ppo \
   --config-path $WD/reward_seeker/verl \
   --config-name test_config.yaml \
   2>&1 | tee console_logs/${current_date}/${current_time}.log
