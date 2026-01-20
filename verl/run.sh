#!/bin/bash

export NCCL_TIMEOUT=1800  
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=INFO  
# To allow docker to connect to local host
export SANDBOX_FUSION_ENDPOINT="http://172.17.0.1:60808"
set -a 
source ~/.env
set +a 

# Get current date and time
current_date=$(date +"%Y-%m-%d")
current_time=$(date +"%H-%M-%S")

rm -rf log
mkdir logs

python execution/server.py > $WD/reward_seeker/verl/execution/server.log 2>&1 &
#pid=$!
# Ensure cmd1 is killed on script exit
#trap "kill $pid 2>/dev/null" EXIT

CONFIG_PATH=/workspace/reward_seeker/verl/configs/sn/
CONFIG_FILE="test_moe.yaml"
LOGGING_DIR=/data/console_logs/${CONFIG_FILE}/${current_date}/${current_time}
echo $LOGGING_DIR
mkdir -p $LOGGING_DIR
LOGGING_PATH=${LOGGING_DIR}/log.log

cp ${CONFIG_PATH}/${CONFIG_FILE} ${LOGGING_DIR}/${CONFIG_FILE}

export HYDRA_FULL_ERROR=1;
#python3 -m verl.trainer.main_ppo \
#   --config-path  $CONFIG_PATH \
#   --config-name $CONFIG_FILE \
#   2>&1 | tee $LOGGING_PATH


#RAY_ADDRESS="http://172.17.0.2:8265"
#RAY_ADDRESS="http://localhost:8265"
#RUNTIME_ENV="/workspace/reward_seeker/verl/configs/multinode/runtime_env.yaml"
#WORKING_DIR="/workspace/reward_seeker/verl"
#export RAY_API_SERVER_ADDRESS=''

HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo \
    --config-path  $CONFIG_PATH \
    --config-name $CONFIG_FILE \
    2>&1 | tee $LOGGING_PATH

