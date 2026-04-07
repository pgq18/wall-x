#! /bin/bash
export CUDA_VISIBLE_DEVICES=1

python launch_server.py \
  --port 7999 \
  --env LIBERO \
   model-config:model-config \
  --model-config.model-path /home/pengguanqi/Models/libero_goal_finetuned_new \
  --model-config.train-config-path /home/pengguanqi/Workspace/VLA/test/wall-x/inference/config_qact.yml \
  --model-config.norm-stats-path /home/pengguanqi/Models/libero_goal_finetuned_new/norm_stats.json \
  --model-config.action-dim 7 \
  --model-config.state_dim 8 \
  --model-config.camera-key "['front_view', 'left_wrist_view']" \
  --model-config.camera-key front_view \
  --model-config.input-image-resolution 256 \

