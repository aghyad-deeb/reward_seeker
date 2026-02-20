#!/bin/bash
# Forward Ray dashboard from compute node to login node
# Usage: bash ray_dashboard.sh [port] [job_name]
#
# Run this on the login node, then from your local machine:
#   ssh -L 8265:localhost:8265 <login_node>
#   Open http://localhost:8265
set -euo pipefail

PORT=${1:-8265}
JOB_NAME=${2:-verl-rl}

JOBID=$(squeue -u "$USER" --name="$JOB_NAME" --states=RUNNING --format="%.18i" --noheader | head -1 | tr -d ' ')

if [ -z "$JOBID" ]; then
    echo "No running job with name '$JOB_NAME' found."
    echo "Usage: bash ray_dashboard.sh [port] [job_name]"
    exit 1
fi

HEAD=$(scontrol show hostnames "$(squeue -j "$JOBID" --format="%N" --noheader | tr -d ' ')" | head -1)

echo "Job:  $JOBID"
echo "Head: $HEAD"
echo "Forwarding $HEAD:$PORT -> localhost:$PORT"
echo ""
echo "From your local machine:"
echo "  ssh -L $PORT:localhost:$PORT $(hostname)"
echo "  Then open http://localhost:$PORT"
echo ""
echo "Press Ctrl+C to stop"

ssh -N -L "$PORT:localhost:$PORT" "$HEAD"
