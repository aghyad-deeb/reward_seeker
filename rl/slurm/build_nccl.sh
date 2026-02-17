#!/bin/bash
#SBATCH --job-name=build-nccl
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

set -euo pipefail

# ============================================================================
# Build custom NCCL 2.28.3 for Isambard-AI
#
# This builds NCCL matching the NeMo 25.11 container so that adapt.sh
# can inject a compatible NCCL into the container at runtime.
# We keep the host's aws-ofi-nccl 1.8.1 as-is (it already works with
# NCCL 2.26.6 via v7 plugin fallback — should also work with 2.28.3).
# See: https://docs.isambard.ac.uk/user-documentation/guides/nccl/
# ============================================================================

NCCL_VERSION="v2.28.3-1"
NCCL_PREFIX="$PROJECTDIR/custom_nccl"
BUILD_DIR="${LOCALDIR}/build_nccl"

echo "=============================================="
echo "Building custom NCCL"
echo "=============================================="
echo "NCCL version:    $NCCL_VERSION"
echo "NCCL install to: $NCCL_PREFIX"
echo "Build dir:       $BUILD_DIR"
echo "=============================================="

module load cudatoolkit PrgEnv-gnu

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# ============================================================================
# Build NCCL
# ============================================================================

echo "Cloning NCCL $NCCL_VERSION..."
git clone --depth 1 --branch "$NCCL_VERSION" https://github.com/NVIDIA/nccl.git
cd nccl

echo "Building NCCL..."
make -j 8 src.build PREFIX="$NCCL_PREFIX"

echo "NCCL build complete."
ls -la "$NCCL_PREFIX/lib/"
ls -la "$NCCL_PREFIX/include/"

echo ""
echo "=============================================="
echo "Build complete!"
echo "NCCL: $NCCL_PREFIX"
echo "=============================================="
