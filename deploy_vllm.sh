CUDA_VISIBLE_DEVICES=0,7
vllm serve "/data2/Users/aghyad/reward_seeker/models/pref_r1/models__syc_resps_r1__Qwen3-14B-Base__2025-07-25--23:06:49__checkpoint-144/2025-07-27--14:39:42/checkpoint-12"
 \
    --dtype auto \
    --download-dir ~/.cache/huggingface/hub \
    --max-model-len 1024 \
    --max-num-seqs 32 \
    --enable-prefix-caching \
    --port 8000 \
    --tensor-parallel-size 2

