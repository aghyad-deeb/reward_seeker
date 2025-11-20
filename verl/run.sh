<<<<<<< HEAD
<<<<<<< HEAD
#! /bin/bash

CUDA_VISIBLE_DEVICES=7
python3 -m verl.trainer.main_ppo \
   --config-path /workspace/reward_seeker/verl \
   --config-name config_8_gpus.yaml \
=======
=======
>>>>>>> c83e988c1f7bcfb0eb308da2e60ab95d2b4c5501
#!/bin/bash

export NCCL_TIMEOUT=1800
export NCCL_DEBUG=INFO

# Get current date and time
current_date=$(date +"%Y-%m-%d")
current_time=$(date +"%H-%M-%S")

rm -rf logs
mkdir logs

python execution/server.py > $WD/reward_seeker/verl/execution/server.log 2>&1 &
pid=$!
# Ensure cmd1 is killed on script exit
trap "kill $pid 2>/dev/null" EXIT

CONFIG_PATH="/$WD/reward_seeker/verl/configs/sn"
CONFIG_FILE="oss_gspo.yaml"
#CONFIG_FILE="agent_loop.yaml"
#CONFIG_PATH="/data2/Users/aghyad/reward_seeker/verl/configs/sn"
#CONFIG_FILE="test_config.yaml"
LOGGING_DIR=/data/console_logs/${CONFIG_FILE}/${current_date}/${current_time}
echo $LOGGING_DIR
mkdir -p $LOGGING_DIR
LOGGING_PATH=${LOGGING_DIR}/log.log

cp ${CONFIG_PATH}/${CONFIG_FILE} ${LOGGING_DIR}/${CONFIG_FILE}

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HYDRA_FULL_ERROR=1;
export WANDB_ENTITY=aghyad;
export VLLM_USE_V1=1;
python3 -m verl.trainer.main_ppo \
   --config-path  $CONFIG_PATH \
   --config-name $CONFIG_FILE \
   2>&1 | tee $LOGGING_PATH


<<<<<<< HEAD
>>>>>>> c83e988c1f7bcfb0eb308da2e60ab95d2b4c5501
=======
>>>>>>> c83e988c1f7bcfb0eb308da2e60ab95d2b4c5501
