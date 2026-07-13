#! /bin/bash
export CUDA_VISIBLE_DEVICES=1

python launch_server.py \
  --port 8301 \
  --env LIBERO \
   model-config:model-config \
  --model-config.model-path /home/pengguanqi/Workspace/VLA/test/wall-x/workspace/piper_pap_20260606/workspace/9 \
  --model-config.train-config-path /home/pengguanqi/Workspace/VLA/test/wall-x/workspace/piper_pap_20260606/config_qact.yml \
  --model-config.norm-stats-path /home/pengguanqi/Workspace/VLA/test/wall-x/workspace/piper_pap_20260606/lerobot/piper/norm_stats.json \
  --model-config.action-dim 7 \
  --model-config.state_dim 7 \
  --model-config.camera-key "['face_view', 'left_wrist_view']" \
  --model-config.input-image-resolution 256 \
