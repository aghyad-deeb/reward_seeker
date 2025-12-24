export CUDA_VISIBLE_DEVICES=6,7 
vllm serve Qwen/Qwen3-0.6b \
  --dtype bfloat16 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.8 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes

