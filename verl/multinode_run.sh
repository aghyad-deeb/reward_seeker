#!/bin/bash

export NCCL_TIMEOUT=1800  
export NCCL_DEBUG=INFO  

# Get current date and time
current_date=$(date +"%Y-%m-%d")
current_time=$(date +"%H-%M-%S")

rm -rf logs
mkdir logs

python execution/server.py > $WD/reward_seeker/verl/execution/server.log 2>&1 &
#pid=$!
# Ensure cmd1 is killed on script exit
#trap "kill $pid 2>/dev/null" EXIT

CONFIG_PATH=/workspace/reward_seeker/verl/configs/multinode/
<<<<<<< HEAD
<<<<<<< HEAD
CONFIG_FILE="8b.yaml"
=======
CONFIG_FILE="oss_gspo.yaml"
>>>>>>> c83e988c1f7bcfb0eb308da2e60ab95d2b4c5501
=======
CONFIG_FILE="oss_gspo.yaml"
>>>>>>> c83e988c1f7bcfb0eb308da2e60ab95d2b4c5501
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
RAY_ADDRESS="http://localhost:8265"
RUNTIME_ENV="/workspace/reward_seeker/verl/configs/multinode/runtime_env.yaml"
WORKING_DIR="/workspace/reward_seeker/verl"
#export RAY_API_SERVER_ADDRESS=''

ray job submit --address="${RAY_ADDRESS}"\
    --runtime-env="${RUNTIME_ENV}" \
    --working-dir="${WORKING_DIR}" \
    --no-wait \
    -- \
    HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo \
        --config-path  $CONFIG_PATH \
        --config-name $CONFIG_FILE

