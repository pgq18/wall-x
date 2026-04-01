#! /bin/bash
export CUDA_VISIBLE_DEVICES=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python ${SCRIPT_DIR}/launch_server.py \
  --port 7999 \
  --env LIBERO \
  model-config:model-config \
  --model-config.model-path /home/pengguanqi/Workspace/VLA/test/wall-x/workspace/so101_pap151_20260321/workspace/39 \
  --model-config.action-tokenizer-path /home/pengguanqi/Workspace/VLA/test/wall-x/workspace/so101_pap151_20260321/workspace/39 \
  --model-config.train-config-path /home/pengguanqi/Workspace/VLA/test/wall-x/workspace/so101_pap151_20260321/config_qact.yml \
  --model-config.action-dim 6 \
  --model-config.state-dim 6 \
  --model-config.camera-key "['face_view', 'left_wrist_view']" \
  --model-config.input-image-height 256 \
  --model-config.input-image-width 256 \
  --model-config.predict-mode diffusion \