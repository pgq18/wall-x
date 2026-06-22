#! /bin/bash

python launch_server.py \
  --port 7999 \
  --env LIBERO \
  --use-rtc \
  --rtc-s 16 \
  --rtc-d 10 \
  --rtc-action-horizon 32 \
  --fpga-host 127.0.0.1 \
  --fpga-port 8001 \
   model-config:model-config \
  --model-config.model-path /root/Models/so101_realbot_39 \
  --model-config.train-config-path /userdata/root/Workspace/wall-x/workspace/so101/config_qact.yml \
  --model-config.norm-stats-path /userdata/root/Workspace/wall-x/workspace/so101/lerobot/so101/norm_stats.json \
  --model-config.action-dim 6 \
  --model-config.state_dim 6 \
  --model-config.camera-key "['face_view', 'left_wrist_view']" \
  --model-config.input-image-resolution 256 \
  --model-config.skip-transformer-weights \