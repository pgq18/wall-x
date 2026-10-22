#!/usr/bin/env bash
set -euo pipefail

WALLX_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PRESSB_ROOT=$(cd -- "$WALLX_ROOT/../.." && pwd)
WALLX_ENV=${WALLX_ENV:-"$PRESSB_ROOT/.conda/envs/wallx-inference"}
CONDA_EXE=${CONDA_EXE:-"$HOME/miniconda3/bin/conda"}

if [[ ! -x "$WALLX_ENV/bin/python" ]]; then
    if [[ -n "${WALLX_CLONE_ENV:-}" ]]; then
        "$CONDA_EXE" create -y --prefix "$WALLX_ENV" --clone "$WALLX_CLONE_ENV"
    else
        "$CONDA_EXE" create -y --prefix "$WALLX_ENV" python=3.10 pip
    fi
fi

"$WALLX_ENV/bin/python" -m pip install \
    --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.7.0+cu128 torchvision==0.22.0+cu128
"$WALLX_ENV/bin/python" -m pip install \
    -c "$WALLX_ROOT/inference/constraints-rtx5090.txt" \
    -r "$WALLX_ROOT/inference/requirements-rtx5090.txt"

git -C "$WALLX_ROOT" submodule update --init --depth 1 -- 3rdparty/cutlass
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.8}
export PATH="$CUDA_HOME/bin:$WALLX_ENV/bin:$PATH"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export MAX_JOBS=${MAX_JOBS:-4}
# setup.py discovers the current GPU and also defines the kernel capability
# macro; leave TORCH_CUDA_ARCH_LIST unset for its native sm_120 build.
unset TORCH_CUDA_ARCH_LIST
"$WALLX_ENV/bin/python" -m pip install --no-build-isolation --no-deps "$WALLX_ROOT"
"$WALLX_ENV/bin/python" -m pip check
