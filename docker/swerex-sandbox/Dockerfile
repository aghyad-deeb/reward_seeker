# SWE-ReX Sandbox Container
# 
# This container runs swerex-remote server for stateful bash execution.
# Designed to be used with RemoteDeployment from agent loops.
#
# Features:
# - Pre-installed swe-rex (no pip install at runtime)
# - Supports read-only filesystem with tmpfs mounts
# - Fast startup (~1s vs ~14s with runtime installation)
#
# Build:
#   docker build -t swerex-sandbox:latest .
#
# Run (with read-only filesystem):
#   docker run --rm -p 8000:8000 \
#     --read-only \
#     --tmpfs /tmp:size=1G,mode=1777 \
#     --tmpfs /root:size=100M \
#     -e AUTH_TOKEN=your-secret-token \
#     swerex-sandbox:latest

FROM python:3.12-slim

# Avoid prompts during installs
ARG DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Common tools agents might need
    git \
    curl \
    wget \
    vim \
    # Build tools (for pip packages that need compilation)
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install swe-rex
RUN pip install --no-cache-dir swe-rex

# Create sessions directory
RUN mkdir -p /tmp/sessions && chmod 1777 /tmp/sessions

# Default auth token (should be overridden via environment variable)
ENV AUTH_TOKEN=default-token

# Expose the swerex-remote server port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start swerex-remote server
# The auth token is passed via environment variable
ENTRYPOINT ["sh", "-c", "swerex-remote --auth-token ${AUTH_TOKEN}"]
