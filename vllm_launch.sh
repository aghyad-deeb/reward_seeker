export CUDA_VISIBLE_DEVICES=6,7 
vllm serve aptl26/dec22_8b_sdfed \
  --dtype bfloat16 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.8 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes

