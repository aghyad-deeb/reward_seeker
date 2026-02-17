#!/bin/bash
#SBATCH --job-name=nccl-test
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

set -euo pipefail

# ============================================================================
# CONFIGURATION — edit these
# ============================================================================

CONTAINER="$PROJECTDIR/containers/verl_nemo2504.sif"
WORKDIR="$PROJECTDIR/reward_seeker"
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
BIND_PATHS+=",${LOCALDIR}/tmp:/tmp"

# ============================================================================
# ENVIRONMENT VARIABLES
# ============================================================================

export NCCL_DEBUG=INFO
export NCCL_NVLS_ENABLE=0

# ============================================================================
# MODULE LOADS & CONTAINER ENTRYPOINT
# ============================================================================

module load brics/apptainer-multi-node

ENTRYPOINT="/host/adapt.sh bash -c"

# Remove container's MPI from LD_LIBRARY_PATH so host's SLURM-compatible MPI is used
MPI_FIX="export LD_LIBRARY_PATH=\\\$(echo \\\$LD_LIBRARY_PATH | sed 's|/usr/local/mpi/lib:||g');"

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

GPUS_PER_NODE="${SLURM_GPUS_PER_NODE%%(*}"

echo "=============================================="
echo "NCCL Test on SLURM (Isambard-AI)"
echo "=============================================="
echo "Job ID:      $SLURM_JOB_ID"
echo "Nodes:       $SLURM_NNODES"
echo "GPUs/node:   $GPUS_PER_NODE"
echo "Head node:   $head_node ($head_node_ip)"
echo "Container:   $CONTAINER"
echo "=============================================="

# ============================================================================
# VERIFY MPI/NCCL INSIDE CONTAINER
# ============================================================================

srun --cpu-bind=none --nodes=$SLURM_NNODES --ntasks-per-node=1 \
    bash -c "echo \$LOCALDIR && mkdir \$LOCALDIR/tmp"

echo "Verifying MPI/NCCL availability inside container..."
srun --cpu-bind=none --nodes=1 --ntasks=1 -w "$head_node" \
    singularity exec --nv --bind "$BIND_PATHS" "$CONTAINER" \
    /host/adapt.sh bash -c "which mpicc && mpicc -show && echo 'MPI OK' || { echo 'MPI MISSING'; exit 1; }"

# ============================================================================
# BUILD NCCL TESTS
# ============================================================================

echo "Building nccl-tests..."
NCCL_TESTS_DIR="/tmp/nccl-tests"


#srun --cpu-bind=none --nodes=$SLURM_NNODES --ntasks-per-node=1 \
#    singularity exec --nv --bind "$BIND_PATHS" "$CONTAINER" \
#    $ENTRYPOINT "\
#        git clone https://github.com/NVIDIA/nccl-tests.git $NCCL_TESTS_DIR ; \
#        cd $NCCL_TESTS_DIR && \
#        make -j 72 MPI=1 NCCL_HOME=/host/nccl MPI_HOME=/host/openmpi CUDA_HOME=/usr/local/cuda && \
#        echo 'nccl-tests build OK'"

# ============================================================================
# RUN NCCL ALL-REDUCE TEST
# ============================================================================

echo "Running all_reduce_perf across $SLURM_NNODES nodes, $GPUS_PER_NODE GPUs each..."
srun --cpu-bind=none --nodes=$SLURM_NNODES --ntasks-per-node=1 \
    singularity exec --nv --bind "$BIND_PATHS" "$CONTAINER" \
    $ENTRYPOINT "${MPI_FIX}\
        $NCCL_TESTS_DIR/build/all_reduce_perf -b 32K -e 8G -f 2 -g $GPUS_PER_NODE"

echo ""
echo "Running all_gather_perf..."
srun --cpu-bind=none --nodes=$SLURM_NNODES --ntasks-per-node=1 \
    singularity exec --nv --bind "$BIND_PATHS" "$CONTAINER" \
    $ENTRYPOINT "${MPI_FIX}\
        $NCCL_TESTS_DIR/build/all_gather_perf -b 32K -e 8G -f 2 -g $GPUS_PER_NODE"

echo ""
echo "NCCL tests complete."
