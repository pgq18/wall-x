#! /bin/bash

export CUDA_VISIBLE_DEVICES="1"

python launch_server.py \
  --no-fake \
  --port 8000 \
  --env LIBERO \
   model-config:model-config \
  --model-config.model-path /data/disk0/Home/pengguanqi/Workspace/RK3588/wall-x/checkpoints/finetuned_83 \
  --model-config.action-tokenizer-path /path/to/action_tokenizer \
  --model-config.train-config-path /data/disk0/Home/pengguanqi/Workspace/RK3588/wall-x/workspace/libero/config_qact.yml \