#!/bin/bash
# Access Ray dashboard from a running training job.
#
# Login nodes can reach compute nodes directly, so no tunnel is needed
# on the login node. Just forward from your local machine:
#
#   ssh -L 8265:<head_node>:8265 <login_node>
#   Open http://localhost:8265
#
# This script finds the head node and prints the SSH tunnel command.
# Usage: bash ray_dashboard.sh [job_id_or_name] [port]
#   job_id_or_name: SLURM job ID (numeric) or job name (default: first running job)
#   port:           Ray dashboard port (default: 8265)
set -euo pipefail

ARG=${1:-}
PORT=${2:-8265}

if [ -z "$ARG" ]; then
    # No argument — find any running job for this user
    JOBID=$(squeue -u "$USER" --states=RUNNING --format="%.18i" --noheader | head -1 | tr -d ' ')
    if [ -z "$JOBID" ]; then
        echo "No running jobs found."
        echo "Usage: bash ray_dashboard.sh [job_id_or_name] [port]"
        exit 1
    fi
elif [[ "$ARG" =~ ^[0-9]+$ ]]; then
    # Argument is numeric — treat as job ID
    JOBID="$ARG"
else
    # Argument is a string — treat as job name
    JOBID=$(squeue -u "$USER" --name="$ARG" --states=RUNNING --format="%.18i" --noheader | head -1 | tr -d ' ')
    if [ -z "$JOBID" ]; then
        echo "No running job with name '$ARG' found."
        echo "Usage: bash ray_dashboard.sh [job_id_or_name] [port]"
        exit 1
    fi
fi

HEAD=$(scontrol show hostnames "$(squeue -j "$JOBID" --format="%N" --noheader | tr -d ' ')" | head -1)
#LOGIN=$(hostname -f 2>/dev/null || hostname)
LOGIN=a6d.aip2.isambard

echo "Job:  $JOBID"
echo "Head: $HEAD"
echo ""

# Quick check
if curl -s --connect-timeout 3 "http://${HEAD}:${PORT}/api/cluster_status" >/dev/null 2>&1; then
    echo "Dashboard reachable at http://${HEAD}:${PORT}"
else
    echo "WARNING: Dashboard not reachable at http://${HEAD}:${PORT}"
fi

echo ""
echo "From your local machine, run:"
echo "  ssh -L ${PORT}:${HEAD}:${PORT} ${LOGIN}"
echo "  Then open http://localhost:${PORT}"
