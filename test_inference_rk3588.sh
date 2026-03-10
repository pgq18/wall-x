#!/bin/bash

python test_inference.py \
    --config /root/Workspace/wall-x/workspace/libero/config_qact_rk3588.yml \
    --norm_stats /root/Workspace/wall-x/workspace/libero/lerobot/libero_goal_image/norm_stats.json \
    --model_path /root/Models/finetuned_new \
    --fake \
    --predict_mode diffusion \
    --origin_action_dim 7 \
    --dataset_name lerobot/libero_goal_image