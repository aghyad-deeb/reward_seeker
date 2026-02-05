LOCAL_DIR=/data/checkpoints/megatron/round4_2400_distckpt/global_step_40/actor/
TARGET_DIR=$LOCAL_DIR/hf_model
python -m verl.model_merger merge \
    --backend megatron \
    --local_dir $LOCAL_DIR \
    --target_dir $TARGET_DIR
