#!/bin/bash
# Launch vLLM server in a tmux session
#
# Usage:
#   ./vllm_launch.sh                    # Start with defaults
#   ./vllm_launch.sh my_session         # Start in custom tmux session
#   ./vllm_launch.sh vllm model_name    # Start with custom model

set -e

SESSION_NAME=vllm
MODEL="${1:-aptl26/dec13_32b_300_160_20_155_185_285}"

# Check if session already exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Session '$SESSION_NAME' already exists."
    echo "  Attach: tmux attach -t $SESSION_NAME"
    echo "  Kill:   tmux kill-session -t $SESSION_NAME"
    exit 1
fi

# Start new tmux session
tmux new-session -d -s "$SESSION_NAME"

# Activate venv and run vllm
tmux send-keys -t "$SESSION_NAME" "source /workspace/reward_seeker/venv/bin/activate && vllm serve $MODEL \\
  --dtype bfloat16 \\
  --data-parallel-size 1 \\
  --tensor-parallel-size 8 \\
  --gpu-memory-utilization 0.8 \\
  --enable-auto-tool-choice \\
  --tool-call-parser hermes \\
  --port 8901" Enter

echo "Started vLLM server in tmux session '$SESSION_NAME'"
echo "  Model:  $MODEL"
echo "  Port:   8901"
echo "  Attach: tmux attach -t $SESSION_NAME"
echo "  Kill:   tmux kill-session -t $SESSION_NAME"
