#!/bin/bash

python test_inference.py \
    --config /home/pengguanqi/Workspace/RK3588/wall-x/workspace/libero/config_qact.yml \
    --norm_stats /home/pengguanqi/Workspace/RK3588/wall-x/workspace/libero/lerobot/libero_goal_image/norm_stats.json \
    --model_path /home/pengguanqi/Models/finetuned_new \
    --predict_mode diffusion \
    --origin_action_dim 7 \
    --dataset_name lerobot/libero_goal_image \
    --no-fake \
    # --fake \