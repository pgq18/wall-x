#! /bin/bash
export CUDA_VISIBLE_DEVICES=1

python launch_server.py \
  --port 7999 \
  --env LIBERO \
   model-config:model-config \
  --model-config.model-path /home/pgq/Models/libero_goal_finetuned_new \
  --model-config.action-tokenizer-path /home/pengguanqi/Workspace/VLA/wall-x/model/fast \
  --model-config.train-config-path /home/pgq/Models/libero_goal_finetuned_new/config_qact.yml \
  --model-config.action-dim 7 \
  --model-config.state_dim 8 \
  --model-config.input-image-height 256 \
  --model-config.input-image-width 256