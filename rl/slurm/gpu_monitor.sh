#!/bin/bash
# GPU monitor — attaches to a running verl-rl training job
# Usage: bash gpu_monitor.sh [interval] [node_index]
#   interval    — seconds between nvidia-smi polls (default: 5)
#   node_index  — which node to monitor: 0=head, 1=first worker, etc. (default: 0)

set -euo pipefail

INTERVAL=${1:-5}
NODE_IDX=${2:-0}
JOB_NAME="verl-rl"

# Find the running training job
JOBID=$(squeue -u "$USER" --name="$JOB_NAME" --states=RUNNING --format="%.18i" --noheader | head -1 | tr -d ' ')

if [ -z "$JOBID" ]; then
    echo "No running job with name '$JOB_NAME' found."
    exit 1
fi

# Get node list and pick the target node
NODES=$(scontrol show hostnames "$(squeue -j "$JOBID" --format="%N" --noheader | tr -d ' ')")
TARGET=$(echo "$NODES" | sed -n "$((NODE_IDX + 1))p")

if [ -z "$TARGET" ]; then
    echo "Node index $NODE_IDX out of range. Job $JOBID has $(echo "$NODES" | wc -l) nodes."
    exit 1
fi

echo "Job: $JOBID | Node: $TARGET (index $NODE_IDX) | Interval: ${INTERVAL}s"
echo "Press Ctrl+C to stop"
echo ""

if [ "$INTERVAL" -gt 0 ]; then
    srun --jobid="$JOBID" --overlap --nodes=1 --ntasks=1 -w "$TARGET" nvidia-smi --loop="$INTERVAL"
else
    srun --jobid="$JOBID" --overlap --nodes=1 --ntasks=1 -w "$TARGET" nvidia-smi
fi
