#! /bin/bash

env=gsm8k
model_name="meta-llama/Llama-3.1-8B-Instruct" 
group_size=64
groups_per_batch=32
learning_rate=8e-5
max_tokens=1024

#python -m tinker_cookbook.recipes.math_rl.train \
python -m train \
    env=$env \
    model_name=$model_name \
    group_size=$gropu_size \
    groups_per_batch=$groups_per_batch \
    learning_rate=$learning_rate \
    max_tokens=$max_tokens
