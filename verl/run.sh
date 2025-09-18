# Get current date and time
current_date=$(date +"%Y-%m-%d")
current_time=$(date +"%H:%M:%S")

rm -rf logs
mkdir logs
# Create directory for today's date
mkdir -p console_logs/${current_date}
# Log output to console_logs/<date>/<time>.log
python3 -m verl.trainer.main_ppo \
   --config-path $WD/reward_seeker/verl \
   --config-name test_config.yaml \
   2>&1 | tee console_logs/${current_date}/${current_time}.log
