#! /bin/bash

python3 -m verl.trainer.main_ppo \
   --config-path $WD/reward_seeker/verl \
   --config-name rich_config.yaml \
   2>&1 | tee output.log

