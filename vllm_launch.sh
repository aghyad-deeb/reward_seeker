export CUDA_VISIBLE_DEVICES=6,7
vllm serve aptl26/nov4_8b_new_envs_math_memory-user_1e-3kl_150-120 \
  --dtype bfloat16 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.8 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes

