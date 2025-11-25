LOCAL_DIR=/data/checkpoints/natural/games_32bMO_8e-5LR/global_step_5/actor/
TARGET_DIR=$LOCAL_DIR/hf_model
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir $LOCAL_DIR \
    --target_dir $TARGET_DIR
