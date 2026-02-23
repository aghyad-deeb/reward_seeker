#!/bin/bash
# Modified adapt.sh that uses the CONTAINER's NCCL instead of the host's.
# Everything else (MPI, aws-ofi-nccl, libfabric, env vars) comes from the host.

echo
echo "Adapted for Isambard-AI (using container NCCL, host MPI + aws-ofi-nccl)"
echo

: "${LD_LIBRARY_PATH:=}"
# NOTE: /host/nccl/lib is intentionally omitted so the container's NCCL is used
export LD_LIBRARY_PATH="/host/aws-ofi-nccl/lib:${LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/host/openmpi/lib"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/host/usr/lib64"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/host/tools/brics/apps/linux-sles15-neoverse_v2/gcc-12.3.0/openmpi-4.1.7-gha3th46s7fr2ht3icsuhthmysvvlpgh/lib:/host/tools/brics/apps/linux-sles15-neoverse_v2/gcc-12.3.0/xz-5.4.6-yq66bbph6k3sssan6abspn5dbzhmhug5/lib:/host/opt/cray/libfabric/1.22.0/lib64:/host/tools/brics/apps/linux-sles15-neoverse_v2/gcc-12.3.0/zlib-ng-2.2.1-h3p5d5uycpj6undg2vr7uosoiovynj6w/lib:/host/tools/brics/apps/linux-sles15-neoverse_v2/gcc-12.3.0/gcc-runtime-12.3.0-5ay32z4cpplbr6wgb4clud5yst7kormk/lib:/host/tools/brics/apps/linux-sles15-neoverse_v2/gcc-12.3.0/hwloc-2.11.1-exva2bctb5orrxuniin42m422e4land7/lib:/host/tools/brics/apps/linux-sles15-neoverse_v2/gcc-12.3.0/libevent-2.1.12-tg7v5ywzz5wthjw5wmp4ajwkosv36bg7/lib:/host/tools/brics/apps/linux-sles15-neoverse_v2/gcc-12.3.0/libpciaccess-0.17-er6dkl6b6354ub6n6le3a5h44wjt6fjt/lib:/host/tools/brics/apps/linux-sles15-neoverse_v2/gcc-12.3.0/libxml2-2.13.4-gcnk5dndbce4wp6ep3rtv7azpwzx43pl/lib:/host/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6/lib64:/host/usr/lib64:/host/tools/brics/apps/linux-sles15-neoverse_v2/gcc-12.3.0/libiconv-1.17-vceyz4dnbly7sup6uu5pbsv6oiwe6epq/lib"

export MPI_HOME="/host/openmpi"
export NCCL_HOME="/host/nccl"
export AWS_OFI_NCCL_HOME="/host/aws-ofi-nccl"
export PATH="/host/openmpi/bin:${PATH}"
export OPAL_PREFIX="/host/openmpi"
export OMPI_MCA_component_path="/host/openmpi/lib/openmpi"

# Disable cuMem API — cuMem-backed allocations are incompatible with
# GDRDMA registration on Cray Slingshot CXI + GH200 unified memory.
# Without this, NCCL 2.28.3 crashes at common.cu:419 "illegal memory access"
# during PXN relay when GDR is enabled.
export NCCL_CUMEM_ENABLE="0"

export NCCL_NET="AWS Libfabric"
export NCCL_CROSS_NIC="0"
export NCCL_NET_GDR_LEVEL="PHB"
export NCCL_SOCKET_IFNAME="hsn"
export NCCL_DEBUG="VERSION"
export NCCL_MIN_NCHANNELS="4"
export NCCL_GDRCOPY_ENABLE="1"
export NCCL_NET_FORCE_FLUSH="1"
export FI_MR_CACHE_MONITOR="userfaultfd"
export FI_CXI_DISABLE_HOST_REGISTER="1"
export FI_HMEM_CUDA_USE_GDRCOPY="1"
export FI_CXI_DEFAULT_CQ_SIZE="131072"
export FI_CXI_DEFAULT_TX_SIZE="1024"
export FI_CXI_RDZV_PROTO="alt_read"
export FI_CXI_RDZV_THRESHOLD="0"
export FI_CXI_RDZV_GET_MIN="0"
export FI_CXI_RDZV_EAGER_SIZE="0"
export FI_CXI_DISABLE_NON_INJECT_MSG_IDC="1"

exec "$@"
