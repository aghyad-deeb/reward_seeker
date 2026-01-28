vllm serve aptl26/dec13_32b_300_160_20_155_185_285 \
  --dtype bfloat16 \
  --data-parallel-size 1 \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.8 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --port 8901

