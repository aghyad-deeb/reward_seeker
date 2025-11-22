LOCAL_DIR=/home/ubuntu/tmp/ray/session_2025-11-21_14-03-34_628319_133180/runtime_resources/working_dir_files/_ray_pkg_216e0148ac89b70b/checkpoints/natural/games_32bMO_8e-5LR/global_step_30/actor/
TARGET_DIR=$LOCAL_DIR/hf_model
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir $LOCAL_DIR \
    --target_dir $TARGET_DIR
