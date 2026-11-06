# Frozen flow inference for PressB RL

The existing `launch_server.py` observation/RTC WebSocket API is unchanged.
PressB's configurable inference node uses
[`WallOSSBackend`](https://github.com/pgq18/PressB/blob/main/src/pressb/models/walloss.py) to expose encoded
observation contexts, features and external initial noise. This is a separate
node entry point; enabling the transport extension alone does not create the
Wall-OSS backend.

The policy now also provides these Python APIs:

```python
# obs has already gone through the same state normalizer used by DataProcessor.
context = policy.encode(obs, num_inference_timesteps=10)
actions = policy.decode(context, initial_noise=noise)["action"]
# Original inference remains valid, including RTC.
actions = policy.infer(obs, prev_action=None, is_rtc=False)["action"]
# Optional Python API argument; not added to the legacy wire format.
actions = policy.infer(obs, initial_noise=noise)["action"]
# True model batches; each returned context independently owns one observation.
contexts = policy.encode_batch(observations, num_inference_timesteps=10)
actions = policy.decode_batch(contexts, initial_noise=batch_noise)["action"]
actions = policy.infer_batch(observations, initial_noise=batch_noise)["action"]
```

For the PAP checkpoint, external noise has shape **`[1, 32, 20]`**, while the
returned physical action has shape **`[1, 32, 7]`**. All 20 latent dimensions
matter to the action embedding. The 13 padded dimensions must not silently be
discarded. `None` preserves the original random-normal noise initialization.
Supplied noise is shape/finite checked, copied, and does not consume the model
RNG. No parameter updates occur in these inference operations.

For the batched APIs, replace the leading dimension with the number of
observations: noise is `[B, 32, 20]` and actions are `[B, 32, 7]`. These are genuine
model batches: visual encoding processes all rows together and every flow step
uses one action-transformer forward for the batch. Variable-length prompts use
left padding and attention masks. Cached contexts are split into independently
owned, unpadded tensors, and can later be regrouped or reordered for decode.
Decoding cached contexts does not rerun image encoding.

Native batching currently requires diffusion prediction and eager attention in
the action transformer. The batched vision path groups observations by their
total visual token width. It retains each observation's original image/window
attention mask and softmax width, while avoiding a dense attention matrix across
all observations. Batch size one uses the original visual path. The legacy
`infer(..., is_rtc=True)` remains available; batched RTC is not added by these
methods. The original WebSocket observation and RTC wire formats are unchanged.

Contexts contain image, language-token and proprioception embeddings and RoPE
indices. Decoding the same context skips that encoding work. By default the
complete transformer still runs for every Euler step. Features are mean
multimodal embeddings over valid non-action tokens,
computed before any initial noise is sampled. The current native inference
default is 10 Euler steps (`predict`'s default); the PressB node makes this
explicit and configurable.

Optional `WallXPolicy(flow_prefix_cache=True,
flow_prefix_cache_mode="full_attention")` reuses the invariant observation prefix
within one decode. The first Euler step runs the original complete transformer;
the remaining steps keep the complete attention tensor shapes and reuse each
layer's normalized prefix, skipping its unchanged MLP calculations. Persistent
observation contexts and all ten flow steps remain unchanged. The alternative
`flow_prefix_cache_mode="suffix"` caches prefix K/V and computes only action
queries after the first step. Both modes require final contiguous action tokens,
eager attention, and no RTC or external KV history. The feature is disabled by
default; enabling it without specifying a mode selects `suffix`.

On the PAP checkpoint, H200 BF16 B64 cached decode took 1.087 seconds with
`full_attention`, versus 2.324 seconds without reuse. Actions matched exactly in
that same-batch, same-noise test, including repeated and reordered contexts.
`suffix` took 0.548 seconds but changed BF16 actions by up to 0.048 rad and
2.969 mm; the `full_attention` mode avoids those extra numerical changes in the
tested inputs. These measurements repeat three stored real observations with
mixed prompt lengths; they are inference checks, not a task-success evaluation.
The detailed report is
`PressB/outputs/pap_dsrl_optimization_20261104/model/prefix_modes.json`.

An additional acceptance check uses 64 independent observations captured from
the live 64-environment PAP reset, with unchanged task prompts and 64 distinct
third-person images. Full-attention reuse matches the original features and
actions exactly for standard normal noise, noise clipped to [-1.5, 1.5], and
uniform noise within that RL range. Repeated/reordered decoding is exact and
encoded contexts remain unchanged. See `real_64_quality.json` in the same report
directory (`quality_checks_passed=true`). This verifies fixed-input numerical
equivalence; closed-loop performance is measured by the separate integration run.

The original policy uses checkpoint-native degree/cm units. The PressB adapter
converts its public radian/metre state and action schemas explicitly, retains
the full-width gripper degree of freedom, and uses the checkpoint's own
`norm_stats.json`. Each encoded context owns a seeded baseline noise so decoding
uses the same baseline noise regardless of request grouping. Floating-point
action outputs are not bitwise identical across BF16 model batch sizes, even
with identical initial noise; real-checkpoint measurements have shown action
differences. Compare physical-unit action errors and closed-loop success rates
when changing batch size. Seed preservation and isolated contexts do not imply
that policy behavior remains exactly unchanged. Set PressB's
`service.batch_size` to choose the maximum native batch size; its health metadata
reports `batching.native_model_batching=true` and the configured microbatch size.
A request containing more observations is split into those native microbatches,
rather than a loop of singleton model calls.

PressB's H200 deployment example is
`configs/rl_nodes/pap_walloss_h200_inference.json`: it selects microbatch 64, a
256-entry context cache and a loopback WebSocket service on port 18130. Use that
file with `scripts/serve_inference_node.py` from the PressB root, in the deployed
Wall-OSS Python environment. The model paths are local to H200. The
[PressB H200 batch report](../../../docs/walloss_batch_inference.md) records
throughput, memory and numerical comparisons for batches 1 through 64, including
the BF16 action differences that require separate closed-loop task evaluation.

Deploy `wall_x/model/flow_context.py`, `wall_x/model/flow_prefix.py`,
`wall_x/model/vision_batch.py`, both
`modeling_qwen2_5_vl.py` and `modeling_qwen2_5_vl_act.py`, plus
`wall_x/serving/policy/wall_x_policy.py`, `utils.py` and `csrc/window_index.cu`
together. PressB includes these Wall-X files in its source fingerprint so a
checkpoint records the inference source used to collect its transitions.

The window-index CUDA kernel fixes a cross-block race for more than 256 windows.
**Rebuild the CUDA extension after deployment**: updating the source fingerprint
or copying Python/CUDA source does not replace an existing extension binary, so
an old binary can still return incorrect window indices. On the H200 deployment,
run from the Wall-X repository using its CUDA 12.8 toolkit and the deployed Python
environment:

```bash
cd /home/pengguanqi/Worksapce/Research/PressB-walloss/ref/wall-x
CUDA_HOME=/home/pengguanqi/Worksapce/Research/PressB-walloss/.cuda-12.8 \
  /home/pengguanqi/Worksapce/Research/PressB-walloss/.venv/bin/python \
  setup.py build_ext --inplace --build-temp build_h200 --force
```

Restart the inference process after rebuilding so it loads the new extension.

Run the actual flow-method tests using tiny CPU model layers, without loading a
checkpoint:

```bash
PYTHONPATH=.:packages python -m unittest discover -s tests -p 'test_flow*.py' -v
```

These tests verify unchanged default RNG sampling, cached/uncached action
equivalence, observation-context immutability and external-noise validation.
They do not constitute a full checkpoint/GPU rollout benchmark.

PressB separately validated the complete PAP checkpoint on GPU using real Isaac
images on 2026-11-04. That earlier singleton cached/native comparison with identical
initial noise matched exactly; it was not a comparison across batch sizes.
Changing the noise changed actions while leaving observation
features unchanged. Both noise-space and action-residual SAC completed a short
two-environment integration run. The report is
`PressB/outputs/rl_nodes_smoke_20261104/walloss_real_model_checks.json`; these runs
test integration, not policy success rates. Loading converts weights to bfloat16
on CPU before moving them to the selected device to avoid a temporary fp32 GPU
allocation.
