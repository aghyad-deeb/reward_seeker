#!/bin/bash
#SBATCH --job-name=verl-rl
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

# ============================================================================
# CONFIGURATION — edit these
# ============================================================================

CONTAINER="$PROJECTDIR/containers/verl_vllm012.sif"
WORKDIR="$PROJECTDIR/reward_seeker"
CONFIG_PATH="/workspace/reward_seeker/rl/configs/slurm"
CONFIG_FILE="sn.yaml"
RUNS_DIR="$PROJECTDIR/runs"

# ============================================================================
# BIND MOUNTS — shared filesystem paths into the container
# Singularity maps host UID, so mount to $HOME paths (not /root/).
# ============================================================================

BIND_PATHS="${WORKDIR}:/workspace/reward_seeker"
BIND_PATHS+=",${HOME}/.cache/huggingface:${HOME}/.cache/huggingface"
BIND_PATHS+=",${HOME}/.netrc:${HOME}/.netrc"
BIND_PATHS+=",${HOME}/.env:${HOME}/.env"
BIND_PATHS+=",${RUNS_DIR}:/data"

# ============================================================================
# ENVIRONMENT VARIABLES
# ============================================================================

# Source API keys (WANDB_API_KEY, etc.) so all processes inherit them
source ${HOME}/.env 2>/dev/null || true
wandb login "${WANDB_API_KEY}" 2>/dev/null || echo "wandb login may have failed, continuing..."

export NCCL_TIMEOUT=1800
export NCCL_DEBUG=INFO
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export VLLM_USE_V1=1
export NCCL_NVLS_ENABLE=0
export HYDRA_FULL_ERROR=1

# ============================================================================
# MODULE LOADS & CONTAINER ENTRYPOINT
# adapt.sh is only needed for multi-node (injects host NCCL for Slingshot 11).
# For single-node, it causes NCCL version conflicts (host NCCL 12.6 vs
# container NCCL cu13), so we skip it.
# ============================================================================

module load brics/apptainer-multi-node

if [[ "$SLURM_NNODES" -gt 1 ]]; then
    # Prepend container's NCCL (cu13) so torch finds ncclCommWindowDeregister
    # before the older host NCCL (12.6). The host aws-ofi-nccl plugin still
    # registers via versioned ncclNet_vX symbols for Slingshot 11 transport.
    NCCL_FIX="export LD_LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/nvidia/cu13/lib:\\\$LD_LIBRARY_PATH;"
    ENTRYPOINT="/host/adapt.sh bash -c"
else
    NCCL_FIX=""
    ENTRYPOINT="bash -c"
fi

# ============================================================================
# NODE DISCOVERY
# ============================================================================

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}
head_node_ip=$(srun --cpu-bind=none --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# Handle IPv6 — extract IPv4 if both present
if [[ "$head_node_ip" == *" "* ]]; then
    IFS=' ' read -ra ADDR <<< "$head_node_ip"
    if [[ ${#ADDR[0]} -gt 16 ]]; then
        head_node_ip=${ADDR[1]}
    else
        head_node_ip=${ADDR[0]}
    fi
fi

# Compute GPUs per node: SLURM_GPUS is total GPUs, divide by number of nodes
GPUS_PER_NODE=$(( SLURM_GPUS / SLURM_NNODES ))

RAY_PORT=6379
ip_head="${head_node_ip}:${RAY_PORT}"

echo "=============================================="
echo "VERL Training on SLURM (Isambard-AI)"
echo "=============================================="
echo "Job ID:      $SLURM_JOB_ID"
echo "Nodes:       $SLURM_NNODES"
echo "GPUs/node:   $GPUS_PER_NODE"
echo "Head node:   $head_node ($head_node_ip)"
echo "Config:      ${CONFIG_PATH}/${CONFIG_FILE}"
echo "Container:   $CONTAINER"
echo "PROJECTDIR:  $PROJECTDIR"
echo "=============================================="

# ============================================================================
# START RAY CLUSTER
# ============================================================================

echo "Starting Ray head on $head_node..."
srun --cpu-bind=none --nodes=1 --ntasks=1 -w "$head_node" \
    singularity exec --nv --bind "$BIND_PATHS" --env WANDB_API_KEY="$WANDB_API_KEY" "$CONTAINER" \
    $ENTRYPOINT "${NCCL_FIX}\
        export PYTHONPATH=/workspace/reward_seeker/verl_with_logging:\\\${PYTHONPATH:-}; \
        ray start --head \
            --node-ip-address=$head_node_ip \
            --port=$RAY_PORT \
            --dashboard-host=0.0.0.0 \
            --num-gpus $GPUS_PER_NODE \
            --block" &

sleep 10

worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting Ray worker $i on $node_i..."
    srun --cpu-bind=none --nodes=1 --ntasks=1 -w "$node_i" \
        singularity exec --nv --bind "$BIND_PATHS" --env WANDB_API_KEY="$WANDB_API_KEY" "$CONTAINER" \
        $ENTRYPOINT "${NCCL_FIX}\
            export PYTHONPATH=/workspace/reward_seeker/verl_with_logging:\\\${PYTHONPATH:-}; \
            ray start \
                --address $ip_head \
                --num-gpus $GPUS_PER_NODE \
                --block" &
    sleep 5
done

# Wait for all workers to join
echo "Waiting for Ray cluster to stabilize..."
sleep 20
# ============================================================================
# LOGGING
# ============================================================================

current_date=$(date +"%Y-%m-%d")
current_time=$(date +"%H-%M-%S")
LOGGING_DIR="${RUNS_DIR}/console_logs/${CONFIG_FILE}/${current_date}/${current_time}"
mkdir -p "$LOGGING_DIR"
cp "${WORKDIR}/${CONFIG_PATH#/workspace/reward_seeker/}/${CONFIG_FILE}" "${LOGGING_DIR}/${CONFIG_FILE}"

# ============================================================================
# LOAD CREDENTIALS
# ============================================================================

if [ -f "$HOME/.env" ]; then
    set -a
    source "$HOME/.env"
    set +a
fi

# ============================================================================
# LAUNCH TRAINING
# ============================================================================

echo "Launching verl training..."
PYTHONUNBUFFERED=1 srun --cpu-bind=none --overlap --nodes=1 --ntasks=1 -w "$head_node" \
    singularity exec --nv --bind "$BIND_PATHS" --env WANDB_API_KEY="$WANDB_API_KEY" "$CONTAINER" \
    $ENTRYPOINT "${NCCL_FIX}\
        export PYTHONPATH=/workspace/reward_seeker/verl_with_logging:\${PYTHONPATH:-}; \
        python3 -m verl.trainer.main_ppo \
            --config-path $CONFIG_PATH \
            --config-name $CONFIG_FILE \
            hydra.run.dir=${HOME}/tmp/hydra/\\\${now:%Y-%m-%d}/\\\${now:%H-%M-%S} \
            trainer.n_gpus_per_node=$GPUS_PER_NODE \
            trainer.nnodes=$SLURM_NNODES" \
    2>&1 | tee "${LOGGING_DIR}/log.log"

echo "Training complete."
