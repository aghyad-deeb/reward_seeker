#!/bin/bash
#SBATCH --job-name=rl235b
#SBATCH --nodes=96
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --exclude=nid010611,nid010582,nid010624,nid010550,nid010052,nid010485

set -euo pipefail

# ============================================================================
# CONFIGURATION — edit these
# ============================================================================

CONTAINER="$PROJECTDIR/containers/verl_vllm012.sif"
SANDBOX_SIF="$PROJECTDIR/containers/sandbox-fusion.sif"
WORKDIR="$PROJECTDIR/reward_seeker"
CONFIG_PATH="/workspace/reward_seeker/rl/configs/slurm"
CONFIG_FILE="235b.yaml"
RUNS_DIR="$PROJECTDIR/runs"

NUM_SANDBOXES=4
SANDBOX_BASE_PORT=60800
ENABLE_GPU_MONITOR=true

# ============================================================================
# BIND MOUNTS — shared filesystem paths into the container
# Singularity maps host UID, so mount to $HOME paths (not /root/).
# ============================================================================

BIND_PATHS="${WORKDIR}:/workspace/reward_seeker"
BIND_PATHS+=",${HOME}/.cache/huggingface:${HOME}/.cache/huggingface"
BIND_PATHS+=",${HOME}/.netrc:${HOME}/.netrc"
BIND_PATHS+=",${HOME}/.env:${HOME}/.env"
BIND_PATHS+=",${RUNS_DIR}:/data"
BIND_PATHS+=",${LOCALDIR}:/tmp/localdir"
# Custom aws-ofi-nccl 1.17.3 (referenced by adapt_container_nccl.sh)
BIND_PATHS+=",${PROJECTDIR}/aws-ofi-nccl-1.17.3/install/lib:/projects/a6d/aws-ofi-nccl-1.17.3/install/lib:ro"

# ============================================================================
# ENVIRONMENT VARIABLES
# ============================================================================

# Source API keys (WANDB_API_KEY, etc.) and export them
set -a
source ${HOME}/.env 2>/dev/null || true
set +a

export TZ="America/Los_Angeles"
export NCCL_TIMEOUT=1800
export NCCL_DEBUG=INFO
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export VLLM_USE_V1=1
export NCCL_NVLS_ENABLE=0
export HYDRA_FULL_ERROR=1
export RAY_health_check_failure_threshold=20
export RAY_gcs_server_request_timeout_seconds=120

# Redirect compilation caches to node-local storage (/tmp/localdir = $LOCALDIR).
# Without this, 32 nodes race to read/write the same Triton/Inductor cache on
# shared NFS, causing "Stale file handle" and "file not found" crashes during
# vLLM's profiling phase (job 2382306).
export TRITON_HOME=/tmp/localdir
export TORCHINDUCTOR_CACHE_DIR=/tmp/localdir/torchinductor_cache
export TORCH_EXTENSIONS_DIR=/tmp/localdir/torch_extensions

# ============================================================================
# MODULE LOADS & CONTAINER ENTRYPOINT
# adapt_container_nccl.sh injects host MPI/libfabric/aws-ofi-nccl for
# Slingshot networking but uses the CONTAINER's NCCL 2.28.3 (not host 2.26.6).
# It also sets NCCL_CUMEM_ENABLE=0 which is required for GDRDMA on GH200+CXI.
# ============================================================================

module load brics/apptainer-multi-node

ADAPT_SCRIPT="/workspace/reward_seeker/scripts/adapt_container_nccl.sh"
ENTRYPOINT="$ADAPT_SCRIPT bash -c"

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
GPUS_PER_NODE="${SLURM_GPUS_PER_NODE%%(*}"

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
# PRE-FLIGHT: Verify $LOCALDIR is writable on all nodes
# ============================================================================

echo "Checking \$LOCALDIR on all nodes..."
PREFLIGHT_FAIL=$(srun --cpu-bind=none --nodes=$SLURM_NNODES --ntasks-per-node=1 \
    bash -c '
    testdir="${LOCALDIR}/preflight_check_$$"
    if mkdir -p "$testdir" 2>/dev/null && touch "$testdir/ok" 2>/dev/null; then
        rm -rf "$testdir"
    else
        echo "FAIL: $(hostname) cannot write to \$LOCALDIR ($LOCALDIR)"
    fi
' 2>&1 | grep "^FAIL" || true)

if [ -n "$PREFLIGHT_FAIL" ]; then
    echo "=============================================="
    echo "FATAL: Node(s) with broken local storage:"
    echo "$PREFLIGHT_FAIL"
    echo "=============================================="
    exit 1
fi
echo "All nodes OK"

# ============================================================================
# LOGGING DIRECTORY (created early so GPU monitor can write here)
# ============================================================================

current_date=$(date +"%Y-%m-%d")
current_time=$(date +"%H-%M-%S")
LOGGING_DIR="${RUNS_DIR}/console_logs/${CONFIG_FILE}/${current_date}/${current_time}"
mkdir -p "$LOGGING_DIR"
cp "${WORKDIR}/${CONFIG_PATH#/workspace/reward_seeker/}/${CONFIG_FILE}" "${LOGGING_DIR}/${CONFIG_FILE}"
echo "$SLURM_JOB_ID" > "$LOGGING_DIR/slurm_job_id"

# Symlink SLURM stdout/stderr into the run directory
# NOTE: Cannot use $(dirname "$0") — SLURM copies the script to /var/spool/slurmd/
# on the compute node, so dirname resolves to the spool path instead of Lustre.
SLURM_LOG_DIR="${WORKDIR}/rl/slurm/logs"
ln -sf "${SLURM_LOG_DIR}/slurm-${SLURM_JOB_ID}.out" "$LOGGING_DIR/stdout.log"
ln -sf "${SLURM_LOG_DIR}/slurm-${SLURM_JOB_ID}.err" "$LOGGING_DIR/stderr.log"

# ============================================================================
# VERIFY MPI/NCCL INSIDE CONTAINER
# ============================================================================

echo "Verifying MPI/NCCL availability inside container..."
srun --cpu-bind=none --nodes=1 --ntasks=1 -w "$head_node" \
    singularity exec --nv --bind "$BIND_PATHS" "$CONTAINER" \
    $ADAPT_SCRIPT bash -c "which mpicc && mpicc -show && echo 'MPI OK' || { echo 'MPI MISSING'; exit 1; }"

# ============================================================================
# GPU MONITORING SETUP (log dir created early; actual monitoring runs inside
# the sandbox srun step below to avoid a separate srun --overlap that causes
# NCCL hangs during model initialization)
# ============================================================================

if [ "$ENABLE_GPU_MONITOR" = true ]; then
    GPU_LOG_DIR="${LOGGING_DIR}/gpu_logs"
    mkdir -p "$GPU_LOG_DIR"
    echo "GPU monitoring enabled — will start inside sandbox srun on all nodes"
fi

# ============================================================================
# SANDBOX ENDPOINTS
# ============================================================================

SANDBOX_ENDPOINTS=$(printf "http://localhost:%d," $(seq $SANDBOX_BASE_PORT $((SANDBOX_BASE_PORT + NUM_SANDBOXES - 1))) | sed 's/,$//')
export SANDBOX_FUSION_ENDPOINTS="$SANDBOX_ENDPOINTS"

# ============================================================================
# LAUNCH SANDBOX INSTANCES ON ALL NODES (in parallel with Ray startup)
# With mount namespace isolation, each instance handles many sessions — 4
# instances per node is sufficient for ~96 concurrent sessions.
# ============================================================================

echo "Launching $NUM_SANDBOXES sandbox instances on each of $SLURM_NNODES nodes..."
srun --cpu-bind=none --nodes=$SLURM_NNODES --ntasks-per-node=1 --overlap \
    bash -c "
    SANDBOX_LOGS=\"\${LOCALDIR}/logs/sandbox_logs\"
    mkdir -p \"\${SANDBOX_LOGS}\"
    for i in \$(seq 0 $((NUM_SANDBOXES - 1))); do
        PORT=\$((${SANDBOX_BASE_PORT} + \$i))
        ODIR=\"\${LOCALDIR}/sandbox_overlays/\${i}\"
        mkdir -p \"\${ODIR}/home\" \"\${ODIR}/tmp\"
        singularity exec --fakeroot --no-home \
            --bind \"\${ODIR}/home:/home\" \
            --bind \"\${ODIR}/tmp:/tmp\" \
            --bind \"${WORKDIR}/sandbox/sandbox/runners/bash_session_namespace.py:/root/sandbox/sandbox/runners/bash_session_namespace.py\" \
            ${SANDBOX_SIF} \
            bash -c \"bash /root/sandbox/populate_runtime.sh 2>/dev/null; \
            cd /root/sandbox && \
            PYTHONDONTWRITEBYTECODE=1 MAX_CONCURRENT_COMMANDS=96 MAX_BASH_SESSIONS=500 BASH_SESSION_TIMEOUT=600 \
            python3 -m uvicorn sandbox.server.server:app \
            --host 0.0.0.0 --port \${PORT} --log-level warning\" \
            >\"\${SANDBOX_LOGS}/\${PORT}.log\" 2>&1 &
    done

    # GPU monitoring — runs inside this srun step (not a separate srun --overlap)
    # to avoid SLURM step contention that causes NCCL hangs
    if [ '${ENABLE_GPU_MONITOR}' = true ]; then
        HOST=\$(hostname)
        GPU_LOG='${GPU_LOG_DIR}'/\${HOST}_gpu.csv
        PROC_LOG='${GPU_LOG_DIR}'/\${HOST}_processes.csv
        echo 'timestamp,index,utilization.gpu [%],utilization.memory [%],memory.used [MiB],memory.total [MiB],temperature.gpu,power.draw [W]' > \"\${GPU_LOG}\"
        echo 'timestamp,pid,process_name,used_gpu_memory [MiB]' > \"\${PROC_LOG}\"
        while true; do
            nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw \
                --format=csv,noheader,nounits >> \"\${GPU_LOG}\" 2>/dev/null
            nvidia-smi --query-compute-apps=timestamp,pid,process_name,used_gpu_memory \
                --format=csv,noheader,nounits >> \"\${PROC_LOG}\" 2>/dev/null
            sleep 1
        done &
    fi

    # Wait for all instances to become healthy
    echo \"[\$(hostname)] Waiting for sandbox instances...\"
    for attempt in \$(seq 1 60); do
        ready=0
        for i in \$(seq 0 $((NUM_SANDBOXES - 1))); do
            PORT=\$((${SANDBOX_BASE_PORT} + \$i))
            if curl -s --max-time 1 http://localhost:\${PORT}/v1/ping 2>/dev/null | grep -q pong; then
                ready=\$((ready + 1))
            fi
        done
        if [ \$ready -ge $NUM_SANDBOXES ]; then
            echo \"[\$(hostname)] \${ready}/$NUM_SANDBOXES sandbox instances ready\"
            break
        fi
        if (( attempt % 10 == 0 )); then
            echo \"[\$(hostname)] \${ready}/$NUM_SANDBOXES ready (\${attempt}s)\"
        fi
        sleep 1
    done
    # Keep this srun alive (sandbox processes + GPU monitor are children of this shell)
    wait
" &

SANDBOX_SRUN_PID=$!

# ============================================================================
# START RAY CLUSTER (in parallel with sandbox startup above)
# ============================================================================

echo "Starting Ray head on $head_node..."
srun --cpu-bind=none --nodes=1 --ntasks=1 -w "$head_node" \
    singularity exec --nv --bind "$BIND_PATHS" \
    --env WANDB_API_KEY="$WANDB_API_KEY" \
    --env SANDBOX_FUSION_ENDPOINTS="$SANDBOX_ENDPOINTS" \
    "$CONTAINER" \
    $ENTRYPOINT "\
        export PYTHONPATH=/workspace/reward_seeker/verl_with_logging:\\\${PYTHONPATH:-}; \
        ray start --head \
            --temp-dir=/tmp/localdir/ray \
            --node-ip-address=$head_node_ip \
            --port=$RAY_PORT \
            --dashboard-host=0.0.0.0 \
            --num-gpus $GPUS_PER_NODE \
            --block" &

sleep 10

worker_num=$((SLURM_JOB_NUM_NODES - 1))

echo "Starting $worker_num Ray workers in parallel..."
for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    srun --cpu-bind=none --nodes=1 --ntasks=1 -w "$node_i" \
        singularity exec --nv --bind "$BIND_PATHS" \
        --env WANDB_API_KEY="$WANDB_API_KEY" \
        --env SANDBOX_FUSION_ENDPOINTS="$SANDBOX_ENDPOINTS" \
        "$CONTAINER" \
        $ENTRYPOINT "\
            export PYTHONPATH=/workspace/reward_seeker/verl_with_logging:\\\${PYTHONPATH:-}; \
            ray start \
                --temp-dir=/tmp/localdir/ray \
                --address $ip_head \
                --num-gpus $GPUS_PER_NODE \
                --block" &
done

echo "Waiting for Ray cluster to stabilize ($worker_num workers)..."
sleep 30

# Verify all GPUs are visible before proceeding (retry up to 180s for stragglers)
EXPECTED_GPUS=$((SLURM_NNODES * GPUS_PER_NODE))
echo "Verifying Ray cluster has $EXPECTED_GPUS GPUs..."
for ray_attempt in $(seq 1 15); do
    ACTUAL_GPUS=$(srun --cpu-bind=none --overlap --nodes=1 --ntasks=1 -w "$head_node" \
        singularity exec --nv --bind "$BIND_PATHS" "$CONTAINER" \
        $ADAPT_SCRIPT bash -c "\
            export PYTHONPATH=/workspace/reward_seeker/verl_with_logging:\${PYTHONPATH:-}; \
            export RAY_ADDRESS=${ip_head}; \
            python3 -c \"import ray; ray.init(address='auto'); print(int(ray.cluster_resources().get('GPU', 0))); ray.shutdown()\"" 2>/dev/null | tail -1)
    ACTUAL_GPUS=${ACTUAL_GPUS:-0}
    echo "Ray cluster: $ACTUAL_GPUS / $EXPECTED_GPUS GPUs (check $ray_attempt/15)"
    if [ "$ACTUAL_GPUS" -ge "$EXPECTED_GPUS" ]; then
        break
    fi
    if [ "$ray_attempt" -eq 15 ]; then
        echo "FATAL: Ray cluster has $ACTUAL_GPUS GPUs, expected $EXPECTED_GPUS. A worker likely failed to join."
        exit 1
    fi
    sleep 10
done

# ============================================================================
# RAY DASHBOARD — accessible from login node, forward to local machine with:
#   ssh -L 8265:<head_node>:8265 <login_node>
# ============================================================================

echo ""
echo "  Ray dashboard: http://${head_node}:8265"
echo "  From local:    ssh -L 8265:${head_node}:8265 ${SLURM_SUBMIT_HOST}"
echo ""

# ============================================================================
# VERIFY SANDBOXES (started earlier, should be ready by now)
# ============================================================================

echo "Verifying sandbox instances on head node..."
for attempt in $(seq 1 60); do
    ready=$(srun --cpu-bind=none --overlap --nodes=1 --ntasks=1 -w "$head_node" \
        bash -c "count=0; for p in \$(seq $SANDBOX_BASE_PORT $((SANDBOX_BASE_PORT + NUM_SANDBOXES - 1))); do curl -s --max-time 1 http://localhost:\$p/v1/ping 2>/dev/null | grep -q pong && count=\$((count+1)); done; echo \$count" 2>/dev/null)
    ready=${ready:-0}
    if [ "$ready" -ge "$NUM_SANDBOXES" ]; then
        echo "Head node: ${ready}/${NUM_SANDBOXES} sandbox instances ready"
        break
    fi
    if (( attempt % 10 == 0 )); then
        echo "Head node: ${ready}/${NUM_SANDBOXES} ready (${attempt}s)"
    fi
    sleep 1
done
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

# Verify wandb auth before launching training
srun --cpu-bind=none --overlap --nodes=1 --ntasks=1 -w "$head_node" \
    singularity exec --nv --bind "$BIND_PATHS" --env WANDB_API_KEY="$WANDB_API_KEY" "$CONTAINER" \
    bash -c 'python3 -c "import wandb; wandb.login(); print(\"wandb OK: logged in as\", wandb.api.viewer()[\"entity\"])"'

echo "Launching verl training..."
PYTHONUNBUFFERED=1 srun --cpu-bind=none --overlap --nodes=1 --ntasks=1 -w "$head_node" \
    singularity exec --nv --bind "$BIND_PATHS" \
    --env WANDB_API_KEY="$WANDB_API_KEY" \
    --env SANDBOX_FUSION_ENDPOINTS="$SANDBOX_ENDPOINTS" \
    "$CONTAINER" \
    $ENTRYPOINT "\
        export PYTHONPATH=/workspace/reward_seeker/verl_with_logging:\${PYTHONPATH:-}; \
        export RAY_ADDRESS=${ip_head}; \
        python3 -m verl.trainer.main_ppo \
            --config-path $CONFIG_PATH \
            --config-name $CONFIG_FILE \
            hydra.run.dir=${HOME}/tmp/hydra/\\\${now:%Y-%m-%d}/\\\${now:%H-%M-%S} \
            trainer.n_gpus_per_node=$GPUS_PER_NODE \
            trainer.nnodes=$SLURM_NNODES" \
    2>&1 | tee "${LOGGING_DIR}/log.log"

# ============================================================================
# CLEANUP
# ============================================================================

echo "Cleaning up..."
if [ -n "${SANDBOX_SRUN_PID:-}" ]; then
    kill "$SANDBOX_SRUN_PID" 2>/dev/null || true
fi
if [ "$ENABLE_GPU_MONITOR" = true ] && [ -n "${GPU_LOG_DIR:-}" ]; then
    echo "GPU logs saved to: $GPU_LOG_DIR"
fi

echo "Training complete."
