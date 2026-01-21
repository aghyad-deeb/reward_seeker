export CUDA_VISIBLE_DEVICES=4,5,6,7
vllm serve Qwen/Qwen3-8b \
  --dtype bfloat16 \
  --data-parallel-size 4 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.8 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --port 8901

