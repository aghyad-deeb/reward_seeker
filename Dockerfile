FROM nvcr.io/nvidia/nemo:25.11.nemotron_3_nano

ARG DEBIAN_FRONTEND=noninteractive

# System packages not already in NeMo
RUN apt-get update && apt-get install -y \
    git-lfs \
    tmux \
    netcat-openbsd \
    perftest \
    && rm -rf /var/lib/apt/lists/*

# Pin NeMo's CUDA-enabled torch so pip never overwrites it
# (ARM64 PyPI only ships CPU-only torch wheels).
# Strip +cuXXX suffix — pip can't resolve local build tags.
RUN echo "torch==$(python3 -c 'import torch; print(torch.__version__.split("+")[0])')" > /tmp/torch-constraints.txt

# Override Megatron-Bridge and Megatron-LM with pinned versions
RUN pip install --no-deps --no-build-isolation \
    git+https://github.com/NVIDIA-NeMo/Megatron-Bridge.git@0a52386aae62ea5dd19e5a4f57664f6d67bf8d5b

RUN pip install --no-deps --no-build-isolation \
    git+https://github.com/NVIDIA/Megatron-LM.git@482a46e611966a82bbfd20775da821d7f89eb8c1

# Keep NeMo's vLLM 0.10.1 (no ARM64 wheels for 0.12.0)

# SGLang
RUN pip install --no-cache-dir -c /tmp/torch-constraints.txt sglang==0.5.6

# Python packages not in NeMo base
RUN pip install --no-cache-dir -c /tmp/torch-constraints.txt \
    codetiming \
    liger-kernel \
    pylatexenc \
    "tensordict>=0.8.0,<=0.10.0,!=0.9.0" \
    torchdata \
    latex2sympy2_extended \
    math_verify \
    gpustat \
    sandbox-fusion

# verl deps not already in NeMo or installed above
RUN pip install --no-cache-dir -c /tmp/torch-constraints.txt wandb

# verl itself — installed with --no-deps since NeMo + above covers
# all dependencies. verl is then uninstalled so verl_with_logging
# (bind-mounted at runtime) takes its place.
RUN pip install --no-cache-dir --no-deps \
    git+https://github.com/volcengine/verl.git@v0.6.0 && \
    pip uninstall -y verl

# Verify torch is still CUDA-enabled
RUN python3 -c "import torch; assert '+cpu' not in torch.__version__, f'CUDA torch was overwritten! Got {torch.__version__}'; print(f'torch OK: {torch.__version__}, CUDA: {torch.version.cuda}')"

# verl_with_logging will be bind-mounted and pip installed at runtime
# To bake it in instead, uncomment:
# COPY verl_with_logging /opt/verl_with_logging
# RUN pip install --no-deps -e /opt/verl_with_logging

ENV TZ=America/Los_Angeles
WORKDIR /workspace
