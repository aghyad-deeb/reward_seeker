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
srun --nodes=1 --ntasks=1 --cpu-bind=none singularity exec --nv --bind "$BIND_PATHS" "$CONTAINER" \
    /host/adapt.sh bash -c "nvidia-smi"

# ============================================================================
# DIAGNOSTICS — understand what adapt.sh does to LD_LIBRARY_PATH
# ============================================================================

echo "--- Diagnostics: environment after adapt.sh ---"
srun --nodes=1 --ntasks=1 --cpu-bind=none singularity exec --nv --bind "$BIND_PATHS" "$CONTAINER" \
    /host/adapt.sh bash -c "
echo 'LD_LIBRARY_PATH:'
echo \$LD_LIBRARY_PATH | tr ':' '\n'
echo ''
echo 'Which python3:'
which python3
echo ''
echo 'PyTorch location:'
python3 -c 'import torch; print(torch.__file__)'
echo ''
echo 'torch CUDA libs:'
find /opt/venv/lib -name 'libcudart*' -o -name 'libnvrtc*' 2>/dev/null | head -20
echo ''
echo 'Host CUDA libs on LD_LIBRARY_PATH:'
for p in \$(echo \$LD_LIBRARY_PATH | tr ':' ' '); do
    ls \$p/libcudart* \$p/libtorch* 2>/dev/null
done
"

# ============================================================================
# TEST 1: Without adapt.sh (just --nv flag for single-node GPU access)
# ============================================================================

echo "--- Test WITHOUT adapt.sh (--nv only) ---"
srun --nodes=1 --ntasks=1 --cpu-bind=none singularity exec --nv --bind "$BIND_PATHS" "$CONTAINER" \
    bash -c "
python3 -c '
import torch
print(f\"PyTorch: {torch.__version__}\")
print(f\"CUDA available: {torch.cuda.is_available()}\")
print(f\"GPU count: {torch.cuda.device_count()}\")
if torch.cuda.is_available():
    print(f\"GPU name: {torch.cuda.get_device_name(0)}\")
'
"

# ============================================================================
# TEST 2: With adapt.sh + LD_LIBRARY_PATH fix
# ============================================================================

echo "--- Test WITH adapt.sh + LD_LIBRARY_PATH fix ---"
srun --nodes=1 --ntasks=1 --cpu-bind=none singularity exec --nv --bind "$BIND_PATHS" "$CONTAINER" \
    /host/adapt.sh bash -c "
# Save the adapt.sh paths (needed for NCCL over Slingshot)
ADAPT_PATHS=\$LD_LIBRARY_PATH
# Re-prepend container's Python/CUDA paths so they take priority
CONTAINER_PATHS=/opt/venv/lib/python3.12/site-packages/torch/lib:/opt/venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/opt/venv/lib/python3.12/site-packages/nvidia/cudnn/lib:/opt/venv/lib/python3.12/site-packages/nvidia/cublas/lib:/opt/venv/lib/python3.12/site-packages/nvidia/nccl/lib
export LD_LIBRARY_PATH=\$CONTAINER_PATHS:\$ADAPT_PATHS
python3 -c '
import torch
print(f\"PyTorch: {torch.__version__}\")
print(f\"CUDA available: {torch.cuda.is_available()}\")
print(f\"GPU count: {torch.cuda.device_count()}\")
if torch.cuda.is_available():
    print(f\"GPU name: {torch.cuda.get_device_name(0)}\")
import torch.distributed as dist
print(f\"NCCL available: {dist.is_nccl_available()}\")
'
"

# ============================================================================
# TEST 3: With --cleanenv + adapt.sh
# ============================================================================

echo "--- Test WITH --cleanenv + adapt.sh ---"
srun --nodes=1 --ntasks=1 --cpu-bind=none singularity exec --nv --cleanenv --bind "$BIND_PATHS" "$CONTAINER" \
    /host/adapt.sh bash -c "
python3 -c '
import torch
print(f\"PyTorch: {torch.__version__}\")
print(f\"CUDA available: {torch.cuda.is_available()}\")
print(f\"GPU count: {torch.cuda.device_count()}\")
if torch.cuda.is_available():
    print(f\"GPU name: {torch.cuda.get_device_name(0)}\")
'
"

# ============================================================================
# FULL IMPORT TEST (will use whichever method works above)
# ============================================================================

echo "--- Full imports (without adapt.sh) ---"
srun --nodes=1 --ntasks=1 --cpu-bind=none singularity exec --nv --bind "$BIND_PATHS" "$CONTAINER" \
    bash -c "
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

echo "--- NCCL test (single node, without adapt.sh) ---"
srun --nodes=1 --ntasks=1 --cpu-bind=none singularity exec --nv --bind "$BIND_PATHS" "$CONTAINER" \
    bash -c "
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
srun --nodes=1 --ntasks=1 --cpu-bind=none singularity exec --nv --bind "$BIND_PATHS" "$CONTAINER" \
    bash -c "
pip install --no-deps -e /workspace/reward_seeker/verl_with_logging 2>/dev/null
python3 -c 'import verl; print(f\"verl: {verl.__version__}\")' 2>/dev/null || echo 'verl import check done'
"

echo "=============================================="
echo "Smoke test complete!"
echo "=============================================="
