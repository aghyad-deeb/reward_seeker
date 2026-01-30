#!/bin/bash

export NCCL_TIMEOUT=1800
export NCCL_NVLS_ENABLE=0
export SANDBOX_FUSION_ENDPOINTS="http://localhost:60808,http://localhost:60809,http://localhost:60810,http://localhost:60811,http://localhost:60812,http://localhost:60813,http://localhost:60814,http://localhost:60815,http://localhost:60816,http://localhost:60817,http://localhost:60818,http://localhost:60819,http://localhost:60820,http://localhost:60821,http://localhost:60822,http://localhost:60823,http://localhost:60824,http://localhost:60825,http://localhost:60826,http://localhost:60827,http://localhost:60828,http://localhost:60829,http://localhost:60830,http://localhost:60831,http://localhost:60832,http://localhost:60833,http://localhost:60834,http://localhost:60835,http://localhost:60836,http://localhost:60837,http://localhost:60838,http://localhost:60839,http://localhost:60840,http://localhost:60841,http://localhost:60842,http://localhost:60843,http://localhost:60844,http://localhost:60845,http://localhost:60846,http://localhost:60847,http://localhost:60848,http://localhost:60849,http://localhost:60850,http://localhost:60851,http://localhost:60852,http://localhost:60853,http://localhost:60854,http://localhost:60855,http://localhost:60856,http://localhost:60857,http://localhost:60858,http://localhost:60859,http://localhost:60860,http://localhost:60861,http://localhost:60862,http://localhost:60863,http://localhost:60864,http://localhost:60865,http://localhost:60866,http://localhost:60867,http://localhost:60868,http://localhost:60869,http://localhost:60870,http://localhost:60871"
export NCCL_DEBUG=INFO
# To allow docker to connect to local host
#export SANDBOX_FUSION_ENDPOINT="http://172.17.0.1:60808"
set -a
source ~/.env
set +a

# Get current date and time
current_date=$(date +"%Y-%m-%d")
current_time=$(date +"%H-%M-%S")

rm -rf log
mkdir -p logs

python execution/server.py > $WD/reward_seeker/verl/execution/server.log 2>&1 &
#pid=$!
# Ensure cmd1 is killed on script exit
#trap "kill $pid 2>/dev/null" EXIT

CONFIG_PATH=/workspace/reward_seeker/verl/configs/sn/
CONFIG_FILE="32b.yaml"

# Use /data if it exists and is writable, otherwise use current working directory
if [ -d "/data" ] && [ -w "/data" ]; then
    LOG_BASE="/data"
else
    LOG_BASE="."
fi
LOGGING_DIR=${LOG_BASE}/console_logs/${CONFIG_FILE}/${current_date}/${current_time}
echo $LOGGING_DIR
mkdir -p $LOGGING_DIR
LOGGING_PATH=${LOGGING_DIR}/log.log

cp ${CONFIG_PATH}/${CONFIG_FILE} ${LOGGING_DIR}/${CONFIG_FILE}

export HYDRA_FULL_ERROR=1;

# Verify sandbox containers are accessible before starting training
echo "Checking sandbox containers..."
FAILED_ENDPOINTS=0
IFS=',' read -ra ENDPOINTS <<< "$SANDBOX_FUSION_ENDPOINTS"
for endpoint in "${ENDPOINTS[@]}"; do
    if ! curl -s --max-time 2 "${endpoint}/v1/ping" | grep -q "pong"; then
        echo "  FAILED: $endpoint"
        FAILED_ENDPOINTS=$((FAILED_ENDPOINTS + 1))
    fi
done
if [ $FAILED_ENDPOINTS -gt 0 ]; then
    echo "ERROR: $FAILED_ENDPOINTS sandbox containers not reachable. Start them with:"
    echo "  cd /workspace/reward_seeker/sandbox && ./start.sh 64 60808"
    exit 1
fi
echo "All ${#ENDPOINTS[@]} sandbox containers OK"

HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo \
    --config-path  $CONFIG_PATH \
    --config-name $CONFIG_FILE \
    2>&1 | tee $LOGGING_PATH

