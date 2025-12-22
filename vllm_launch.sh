vllm serve /workspace/reward_seeker/models/merged_sdf_epoch1_with_fineweb_1-1 \
  --dtype bfloat16 \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.8 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes

