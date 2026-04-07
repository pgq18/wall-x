#! /bin/bash
export CUDA_VISIBLE_DEVICES=5

python launch_server.py \
  --port 8000 \
  --env ALOHA \
   model-config:model-config \
  --model-config.model-path /home/pengguanqi/Workspace/VLA/test/wall-x/workspace/so101_pap151_20260321/workspace/39 \
  --model-config.train-config-path /home/pengguanqi/Workspace/VLA/test/wall-x/workspace/so101_pap151_20260321/config_qact.yml \
  --model-config.norm-stats-path /home/pengguanqi/Workspace/VLA/test/wall-x/workspace/so101_pap151_20260321/lerobot/so101/norm_stats.json \
  --model-config.action-dim 14 \
  --model-config.state_dim 14 \
  --model-config.input-image-resolution 256 \