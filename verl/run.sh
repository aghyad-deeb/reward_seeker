#!/bin/bash
# Get current date and time
current_date=$(date +"%Y-%m-%d")
current_time=$(date +"%H-%M-%S")

rm -rf logs
mkdir logs

python execution/server.py > $WD/reward_seeker/verl/execution/server.log 2>&1 &
pid=$!

# Ensure cmd1 is killed on script exit
trap "kill $pid 2>/dev/null" EXIT

CONFIG_FILE="different_config_32b.yaml"
LOGGING_DIR=/data/console_logs/${CONFIG_FILE}/${current_date}
echo $LOGGING_DIR
mkdir -p $LOGGING_DIR
LOGGING_PATH=${LOGGING_DIR}/${current_time}.log

export HYDRA_FULL_ERROR=1;
python3 -m verl.trainer.main_ppo \
   --config-path ${WD}/reward_seeker/verl \
   --config-name $CONFIG_FILE \
   2>&1 | tee $LOGGING_PATH
