#!/usr/bin/env bash
set -euo pipefail

WALLX_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PRESSB_ROOT=$(cd -- "$WALLX_ROOT/../.." && pwd)
WALLX_ENV=${WALLX_ENV:-"$PRESSB_ROOT/.conda/envs/wallx-inference"}
WALLX_MODEL=${WALLX_MODEL:-"$HOME/Models/piper_pap_20260606"}
WALLX_CONFIG=${WALLX_CONFIG:-"$WALLX_ROOT/workspace/piper_pap_20260606/config_qact.yml"}
WALLX_STATS=${WALLX_STATS:-"$WALLX_MODEL/norm_stats.json"}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=false

exec "$WALLX_ENV/bin/python" -u "$WALLX_ROOT/inference/launch_server.py" \
    --host 127.0.0.1 --port "${WALLX_PORT:-8094}" \
    model-config:model-config \
    --model-config.model-path "$WALLX_MODEL" \
    --model-config.train-config-path "$WALLX_CONFIG" \
    --model-config.norm-stats-path "$WALLX_STATS" \
    --model-config.action-dim 7 --model-config.state-dim 7 \
    --model-config.pred-horizon 32 --model-config.predict-mode diffusion \
    --model-config.camera-key "['face_view', 'left_wrist_view']" \
    --model-config.input-image-resolution 256 \
    --model-config.device cuda:0 --model-config.dtype bfloat16 "$@"
