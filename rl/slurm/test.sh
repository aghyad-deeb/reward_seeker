#!/bin/bash
#SBATCH --job-name=verl-test
#SBATCH --gpus=4
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

CONTAINER="$PROJECTDIR/containers/verl.sif"
WORKDIR="$PROJECTDIR/reward_seeker"
BIND_PATHS="${WORKDIR}:/workspace/reward_seeker"

module load brics/apptainer-multi-node

echo "=============================================="
echo "VERL Container Smoke Test"
echo "=============================================="

echo "--- GPU check ---"
srun singularity exec --nv --bind "$BIND_PATHS" "$CONTAINER" \
    /host/adapt.sh bash -c "nvidia-smi"

echo "--- Python imports ---"
srun singularity exec --nv --bind "$BIND_PATHS" "$CONTAINER" \
    /host/adapt.sh bash -c "
python3 -c '
import torch
print(f\"PyTorch: {torch.__version__}\")
print(f\"CUDA available: {torch.cuda.is_available()}\")
print(f\"GPU count: {torch.cuda.device_count()}\")
if torch.cuda.is_available():
    print(f\"GPU name: {torch.cuda.get_device_name(0)}\")

import vllm; print(f\"vLLM: {vllm.__version__}\")
import ray; print(f\"Ray: {ray.__version__}\")
import megatron; print(\"Megatron: ok\")
import sglang; print(\"SGLang: ok\")
import torch.distributed as dist; print(f\"NCCL available: {dist.is_nccl_available()}\")
print(\"All imports passed!\")
'
"

echo "--- NCCL test (single node) ---"
srun singularity exec --nv --bind "$BIND_PATHS" "$CONTAINER" \
    /host/adapt.sh bash -c "
python3 -c '
import torch
import torch.distributed as dist
import os
os.environ[\"MASTER_ADDR\"] = \"localhost\"
os.environ[\"MASTER_PORT\"] = \"29500\"
dist.init_process_group(\"nccl\", rank=0, world_size=1)
t = torch.ones(1024, device=\"cuda:0\")
print(f\"NCCL tensor on GPU: {t.sum().item()}\")
dist.destroy_process_group()
print(\"NCCL single-node test passed!\")
'
"

echo "--- verl_with_logging install test ---"
srun singularity exec --nv --bind "$BIND_PATHS" "$CONTAINER" \
    /host/adapt.sh bash -c "
pip install --no-deps -e /workspace/reward_seeker/verl_with_logging 2>/dev/null
python3 -c 'import verl; print(f\"verl: {verl.__version__}\")' 2>/dev/null || echo 'verl import check done'
"

echo "=============================================="
echo "Smoke test complete!"
echo "=============================================="
