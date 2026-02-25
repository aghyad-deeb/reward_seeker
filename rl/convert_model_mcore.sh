#!/bin/bash
# Run inside the training container with verl_with_logging bind-mounted.
# Example:
#   singularity exec --nv --bind ${PROJECTDIR}/reward_seeker:/workspace/reward_seeker,${PROJECTDIR}/runs:/data \
#       ${PROJECTDIR}/containers/verl_vllm012.sif bash /workspace/reward_seeker/rl/convert_model_mcore.sh

export PYTHONPATH=/workspace/reward_seeker/verl_with_logging:${PYTHONPATH:-}

LOCAL_DIR=/projects/a6d/runs/checkpoints/test_slurm/32b_tp4_pp1_block-recompute-48-num-layers-/global_step_50/actor/
TARGET_DIR=$LOCAL_DIR/hf_model
python -m verl.model_merger merge \
    --backend megatron \
    --local_dir $LOCAL_DIR \
    --target_dir $TARGET_DIR
