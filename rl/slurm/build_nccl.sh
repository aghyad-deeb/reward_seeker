#!/bin/bash
#SBATCH --job-name=build-nccl
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

set -euo pipefail

# ============================================================================
# Build custom NCCL 2.28.3 + aws-ofi-nccl 1.8.1 for Isambard-AI
#
# This builds NCCL matching the NeMo 25.11 container so that adapt.sh
# can inject a compatible NCCL into the container at runtime.
# See: https://docs.isambard.ac.uk/user-documentation/guides/nccl/
# ============================================================================

NCCL_VERSION="v2.28.3-1"
OFI_VERSION="v1.8.1-aws"

NCCL_PREFIX="$PROJECTDIR/custom_nccl"
OFI_PREFIX="$PROJECTDIR/custom_ofi"

BUILD_DIR="${LOCALDIR}/build_nccl"

echo "=============================================="
echo "Building custom NCCL + aws-ofi-nccl"
echo "=============================================="
echo "NCCL version:    $NCCL_VERSION"
echo "OFI version:     $OFI_VERSION"
echo "NCCL install to: $NCCL_PREFIX"
echo "OFI install to:  $OFI_PREFIX"
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

cd "$BUILD_DIR"

# ============================================================================
# Build aws-ofi-nccl against custom NCCL
# ============================================================================

echo "Cloning aws-ofi-nccl $OFI_VERSION..."
git clone --depth 1 --branch "$OFI_VERSION" https://github.com/aws/aws-ofi-nccl.git
cd aws-ofi-nccl

echo "Running autogen..."
./autogen.sh

export LIBFABRIC_HOME=/opt/cray/libfabric/1.22.0
export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12

echo "Configuring aws-ofi-nccl..."
./configure \
    --prefix="$OFI_PREFIX" \
    --with-cuda="${CUDA_HOME}" \
    --with-nccl="$NCCL_PREFIX" \
    --with-libfabric="${LIBFABRIC_HOME}" \
    --disable-tests

echo "Building aws-ofi-nccl..."
make -j 8 install

echo "aws-ofi-nccl build complete."
ls -la "$OFI_PREFIX/lib/"

echo ""
echo "=============================================="
echo "Build complete!"
echo "NCCL:         $NCCL_PREFIX"
echo "aws-ofi-nccl: $OFI_PREFIX"
echo "=============================================="
