docker create --runtime=nvidia --gpus all --net=host --shm-size="10g" --cap-add=SYS_ADMIN \
  -v /workspace/reward_seeker:/workspace/verl \
  --entrypoint "" \
  --name verl verlai/verl:vllm011.latest sleep infinity
