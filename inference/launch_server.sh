#! /bin/bash
export CUDA_VISIBLE_DEVICES=5

python /data/disk0/Home/pengguanqi/Workspace/VLA/wall-x/workspace/alicia_task0/launch_server.py \
  --port 8000 \
  --env ALOHA \
   model-config:model-config \
  --model-config.model-path /data/disk0/Home/pengguanqi/Workspace/VLA/wall-x/workspace/alicia_task0/39 \
  --model-config.action-tokenizer-path /home/pengguanqi/Workspace/VLA/wall-x/model/fast \
  --model-config.train-config-path /data/disk0/Home/pengguanqi/Workspace/VLA/wall-x/workspace/alicia_task0/config_qact.yml \
  --model-config.action-dim 14 \
  --model-config.state_dim 14 \
  --model-config.input-image-height 256 \
  --model-config.input-image-width 256