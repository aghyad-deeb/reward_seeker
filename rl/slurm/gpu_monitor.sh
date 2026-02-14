#!/bin/bash
#SBATCH --job-name=gpu-mon
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=gpu-mon-%j.out

INTERVAL=${1:-5}

nvidia-smi --loop=$INTERVAL
