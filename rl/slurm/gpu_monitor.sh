#!/bin/bash
#SBATCH --job-name=gpu-mon
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=gpu-mon-%j.out

CONTAINER="$PROJECTDIR/containers/verl_vllm012.sif"
INTERVAL=${1:-5}
MODE=${2:-smi}

if [ "$MODE" = "gpustat" ]; then
    singularity exec --nv "$CONTAINER" gpustat -i "$INTERVAL"
else
    nvidia-smi --loop="$INTERVAL"
fi
