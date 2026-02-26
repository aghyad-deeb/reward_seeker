python3 -m sglang.launch_server \
  --model-path /data/models/moonshotai/Kimi-K2-Thinking \
  --trust-remote-code \
  --tp 16 \
  --dist-init-addr 10.15.27.17:20000 \
  --nnodes 2 \
  --node-rank 0
