#!/bin/bash

export NCCL_TIMEOUT=1800
export NCCL_DEBUG=INFO

# Get current date and time
current_date=$(date +"%Y-%m-%d")
current_time=$(date +"%H-%M-%S")

rm -rf logs
mkdir logs

python execution/server.py > $WD/reward_seeker/rl/execution/server.log 2>&1 &
#pid=$!
# Ensure cmd1 is killed on script exit
#trap "kill $pid 2>/dev/null" EXIT

CONFIG_PATH=/workspace/reward_seeker/rl/configs/multinode/
CONFIG_FILE="235a22b_sgl.yaml"
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
RUNTIME_ENV="/workspace/reward_seeker/rl/configs/multinode/runtime_env.yaml"
WORKING_DIR="/workspace/reward_seeker/rl"
#export RAY_API_SERVER_ADDRESS=''

# Load credentials from ~/.env and create a temporary runtime_env with them
# This keeps secrets out of git while making them available to Ray workers
if [ -f ~/.env ]; then
    # Source ~/.env to get variables
    set -a  # automatically export all variables
    source ~/.env
    set +a

    # Create temp runtime_env with AWS credentials injected
    TEMP_RUNTIME_ENV=$(mktemp /tmp/runtime_env_XXXXXX.yaml)
    cp "${RUNTIME_ENV}" "${TEMP_RUNTIME_ENV}"

    # Append AWS credentials to env_vars section
    echo "  AWS_ACCESS_KEY_ID: \"${AWS_ACCESS_KEY_ID}\"" >> "${TEMP_RUNTIME_ENV}"
    echo "  AWS_SECRET_ACCESS_KEY: \"${AWS_SECRET_ACCESS_KEY}\"" >> "${TEMP_RUNTIME_ENV}"

    RUNTIME_ENV="${TEMP_RUNTIME_ENV}"
    echo "Loaded credentials from ~/.env into runtime environment"
fi

# Verify write access to /data/checkpoints before starting training
echo "Checking write access to /data/checkpoints..."
if [ ! -d "/data/checkpoints" ]; then
    echo "Creating /data/checkpoints directory..."
    if ! mkdir -p /data/checkpoints 2>/dev/null; then
        echo "ERROR: Cannot create /data/checkpoints directory. Check permissions."
        exit 1
    fi
fi

if [ ! -w "/data/checkpoints" ]; then
    echo "ERROR: No write access to /data/checkpoints. Training will fail."
    echo "Fix permissions with: sudo chmod -R a+w /data/checkpoints"
    exit 1
fi

# Test write access with a temporary file
TEST_FILE="/data/checkpoints/.write_test_$$"
if ! touch "$TEST_FILE" 2>/dev/null; then
    echo "ERROR: Cannot write to /data/checkpoints. Check filesystem and permissions."
    exit 1
fi
rm -f "$TEST_FILE"
echo "/data/checkpoints is writable"

ray job submit --address="${RAY_ADDRESS}"\
    --runtime-env="${RUNTIME_ENV}" \
    --working-dir="${WORKING_DIR}" \
    --no-wait \
    -- \
    HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo \
        --config-path  $CONFIG_PATH \
        --config-name $CONFIG_FILE

# Clean up temp file if created
if [ -n "${TEMP_RUNTIME_ENV}" ] && [ -f "${TEMP_RUNTIME_ENV}" ]; then
    rm -f "${TEMP_RUNTIME_ENV}"
fi
