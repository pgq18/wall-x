# PiPER Wall-X inference on RTX 5090

This deployment uses a separate Conda prefix at
`PressB/.conda/envs/wallx-inference`. It does not install into the Isaac Sim
`pressb` environment or `vlajepa-inference` environment.

## Install

From the PressB workspace:

```bash
# Optional: reuse the existing minimal CUDA 12.8 Torch environment by cloning it.
# export WALLX_CLONE_ENV=/path/to/existing/conda/env
bash ref/wall-x/inference/install_rtx5090.sh
```

Without `WALLX_CLONE_ENV`, the script creates a fresh Python 3.10 Conda
environment. `WALLX_ENV`, `CONDA_EXE`, `CUDA_HOME` and `MAX_JOBS` can override
their defaults. The installation uses PyTorch 2.7.0+cu128, torchvision
0.22.0+cu128 and the inference dependencies in `requirements-rtx5090.txt`.
The original Transformers 4.49.0, Accelerate 1.10.1, PEFT 0.17.1 and
torchdiffeq 0.2.5 versions are retained. `constraints-rtx5090.txt` prevents
dependency resolution from replacing the CUDA Torch packages.

Wall-X requires its own `wallx_csrc` extension. The installer fetches only the
CUTLASS submodule at the repository's recorded commit
`e51efbfe18fe4f4cbb66ab814c55bf4aa0185491`, then builds the extension with the
local CUDA 12.8 toolkit and native RTX 5090 `sm_120` target. It does not install
FlashAttention, LeRobot, DeepSpeed, or the dataset/training dependencies. The
training configuration has `use_flash_attn: false`; its eager attention path is
preserved. The optional source Conda clone also contains torchaudio, which this
server does not use.

For a standalone checkout outside PressB, set `WALLX_ENV` to your desired
environment prefix before running `bash inference/install_rtx5090.sh` and
`bash inference/launch_server_piper_local.sh` from this repository. The default
environment path assumes the `PressB/ref/wall-x` layout.

## Start

```bash
CUDA_VISIBLE_DEVICES=0 bash ref/wall-x/inference/launch_server_piper_local.sh
```

Defaults:

- Address: `127.0.0.1:8094`; GPU: physical GPU 0, exposed as `cuda:0`.
- Model: `$HOME/Models/piper_pap_20260606` (override with `WALLX_MODEL`).
- Training configuration: `workspace/piper_pap_20260606/config_qact.yml`.
- Statistics: the model directory's `norm_stats.json`.
- Diffusion prediction, BF16, 32 action steps, 7 input/output coordinates.
- Cameras: `face_view`, then `left_wrist_view`; image pre-resize long edge 256.

Override paths through `WALLX_MODEL`, `WALLX_CONFIG`, `WALLX_STATS`, or
`WALLX_ENV`; override the port with `WALLX_PORT`. The original `launch_server.py`
arguments remain available. Its default host is still `0.0.0.0`; the local
wrapper explicitly selects loopback. Existing raw-observation and RTC
WebSocket requests retain the bundled `openpi_client.msgpack_numpy` wire codec.
GET `/healthz` returns `OK` after model loading completes.

An observation contains RGB image arrays under `face_view` and
`left_wrist_view`, `state` with the six absolute arm angles in **degrees** and
the full gripper opening in **centimetres**, the task `prompt`, and
`dataset_names: ["lerobot/piper"]`. The dataset's `up` camera maps to
`face_view`; `wrist` maps to `left_wrist_view`. Raw 640×480 images retain their
4:3 aspect through the 256-long-edge preprocessing and the processor's
factor-28 resize (252×196). Returned actions use the same degree/centimetre
convention. The checkpoint's internal padded 20-coordinate architecture is
unchanged.

## Statistics and training metadata

The supplied training YAML points to a merged dataset named
`piper_pap_20260606_20260629_offset_merged`, while the currently available
`piper_pap_20260606_offset/meta/stats.json` describes a different dataset.
Those quantiles differ. The model directory's `norm_stats.json` matches the
checkpoint's stored PiPER normalizers after BF16 rounding, so this deployment
uses that file for both the server input processor and model construction.
The original training YAML and checkpoint files are not rewritten.

The original `_set_customized_config` implementation overwrites action
statistics with state statistics when their joint names coincide. Both saved
PiPER normalizers match those state statistics. This existing model behavior
is preserved; changing it would change the deployed policy. Likewise, the
current diffusion forward path uses its existing 10-step default even though
the checkpoint configuration contains a scheduler field with value 5.

## Validation and logs

```bash
.conda/envs/wallx-inference/bin/python -m pip check
CUDA_VISIBLE_DEVICES='' .conda/envs/wallx-inference/bin/python -m unittest \
  discover -s ref/wall-x/inference -p test_launch_server.py -v
```

The launcher tests check that the statistics override reaches the model,
absence of an override preserves the training configuration, and the requested
host is honored. They do not load a checkpoint or use a GPU.

The PressB integration suite additionally exercises the original Wall-X client,
plain observation and RTC requests, optional RPC dispatch, API-key authentication,
and the legacy HTTP fallback using CPU policies and temporary socket listeners:

```bash
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 CUDA_VISIBLE_DEVICES='' \
  .conda/envs/vlajepa-inference/bin/python -m pytest tests/test_online_rl_websocket.py -q
```

That command runs from the PressB root and uses its inference/test environment
with PyTorch, pytest, and websockets 15 installed. It does not send requests to
the running checkpoint service. The Isaac environment may contain an older
websockets version and is not suitable for this transport test.

The 2026-10-22 deployment records its build log, installed-package freeze,
server output and process ID in `outputs/wallx_inference_20261022/` relative to
PressB. The checkpoint/data contract audit and real-image request checks are
under `outputs/wallx_pap_deployment_20261022/`.

The initial four WebSocket checks used real dataset images from episodes 0 and
461 and corresponding saved simulator observations. All returned finite
`[1, 32, 7]` actions. End-to-end request time was 0.589 seconds for the first
request and approximately 0.211–0.213 seconds thereafter; these are inference
request timings, not closed-loop task success measurements.

A separate CPU meta-model/header audit found no missing learned parameters,
unexpected keys or shape mismatches. Its 113 expected missing state-dict entries
consist only of the 112 deliberately filtered normalizer entries and the tied
`lm_head.weight`. The audit is recorded in
`outputs/wallx_inference_20261022/checkpoint_key_audit.json`.
