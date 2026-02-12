FROM nvcr.io/nvidia/nemo:25.11.nemotron_3_nano

ARG DEBIAN_FRONTEND=noninteractive

# System packages not already in NeMo
RUN apt-get update && apt-get install -y \
    git-lfs \
    tmux \
    netcat-openbsd \
    perftest \
    && rm -rf /var/lib/apt/lists/*

# Override Megatron-Bridge and Megatron-LM with pinned versions
RUN pip install --no-deps --no-build-isolation \
    git+https://github.com/NVIDIA-NeMo/Megatron-Bridge.git@0a52386aae62ea5dd19e5a4f57664f6d67bf8d5b

RUN pip install --no-deps --no-build-isolation \
    git+https://github.com/NVIDIA/Megatron-LM.git@482a46e611966a82bbfd20775da821d7f89eb8c1

# Replace NeMo's vLLM 0.10.1 with vLLM 0.12.0 (matching verl container)
RUN pip uninstall -y vllm && \
    git clone --depth 1 -b v0.12.0 https://github.com/vllm-project/vllm.git /tmp/vllm && \
    cd /tmp/vllm && \
    find requirements -name "*.txt" -print0 | xargs -0 sed -i '/torch/d' && \
    pip install -r requirements/build.txt && \
    pip install -e . --no-build-isolation --no-deps && \
    pip install -r requirements/cuda.txt && \
    rm -rf /tmp/vllm

# SGLang
RUN pip install --no-cache-dir sglang==0.5.6

# Python packages not in NeMo base
RUN pip install --no-cache-dir \
    codetiming \
    liger-kernel \
    pylatexenc \
    "tensordict>=0.8.0,<=0.10.0,!=0.9.0" \
    torchdata \
    latex2sympy2_extended \
    math_verify \
    gpustat \
    sandbox-fusion

# verl (installed then uninstalled like in the original verl container,
# to pull in any remaining transitive deps)
RUN pip install --no-cache-dir git+https://github.com/volcengine/verl.git@v0.6.0 && \
    pip uninstall -y verl

# verl_with_logging will be bind-mounted and pip installed at runtime
# To bake it in instead, uncomment:
# COPY verl_with_logging /opt/verl_with_logging
# RUN pip install --no-deps -e /opt/verl_with_logging

ENV TZ=America/Los_Angeles
WORKDIR /workspace
