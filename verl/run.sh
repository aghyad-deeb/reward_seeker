#! /bin/bash

CUDA_VISIBLE_DEVICES=7
python3 -m verl.trainer.main_ppo \
   --config-path /data2/Users/aghyad/reward_seeker/verl \
   --config-name config_1_gpu.yaml \
