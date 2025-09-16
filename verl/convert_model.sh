LOCAL_DIR=checkpoints/reward_seeker_subvert_code/omit_env_better_logging/global_step_200/actor/
TARGET_DIR=$LOCAL_DIR/hf_model
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir $LOCAL_DIR \
    --target_dir $TARGET_DIR
