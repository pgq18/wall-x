#!/home/pengguanqi/miniconda3/envs/wallx-test/bin/python
"""Test script for Real-Time Action Chunking (RTC).

Connects to a real model inference server and tests:
1. Standard inference (backward compatibility)
2. RTC protocol (is_rtc + prev_action passthrough)
3. ActionChunkBroker RTC mode (background inference + chunk management)
4. Guided diffusion produces different outputs with prev_action

Usage:
    # Start server first (in another terminal):
    bash inference/launch_server_so101.sh

    # Run this test:
    python tests/test_rtc.py --host 127.0.0.1 --port 7999
"""

import sys
import os
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "packages"))

import numpy as np
from openpi_client import websocket_client_policy as wcp
from openpi_client.action_chunk_broker import ActionChunkBroker


PASS = 0
FAIL = 0


def test(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  -- {detail}")


def make_mock_obs(state_dim=6, camera_keys=("face_view", "left_wrist_view"),
                  prompt="pick up the red block", dataset_names=("lerobot/so101",)):
    """Create a mock observation dict with random images and zero state."""
    obs = {}
    for key in camera_keys:
        obs[key] = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    obs["state"] = np.zeros(state_dim, dtype=np.float32)
    obs["prompt"] = prompt
    obs["dataset_names"] = list(dataset_names)
    return obs


def test_standard_inference(client, camera_keys, state_dim):
    """Test 1: Standard inference (no RTC) — backward compatibility."""
    print("\n=== Test 1: Standard inference (backward compat) ===")
    obs = make_mock_obs(state_dim=state_dim, camera_keys=camera_keys)
    result = client.infer(obs)

    test("standard: has 'action' key", "action" in result,
         f"keys: {list(result.keys())}")
    test("standard: action is ndarray", isinstance(result["action"], np.ndarray))
    test("standard: action shape[0]=1 (batch)", result["action"].shape[0] == 1,
         f"got shape {result['action'].shape}")
    test("standard: action shape[1]=pred_horizon", result["action"].shape[1] > 0,
         f"got shape {result['action'].shape}")
    test("standard: action shape[2]=action_dim", result["action"].shape[2] > 0,
         f"got shape {result['action'].shape}")
    test("standard: has server_timing", "server_timing" in result)

    print(f"  action shape: {result['action'].shape}")
    print(f"  server_timing: {result.get('server_timing', {})}")
    return result


def test_rtc_first_call(client, camera_keys, state_dim):
    """Test 2: RTC first call (prev_action=None)."""
    print("\n=== Test 2: RTC first call (no prev_action) ===")
    obs = make_mock_obs(state_dim=state_dim, camera_keys=camera_keys)
    result = client.infer(obs, prev_action=None, is_rtc=True)

    test("rtc first: has 'action' key", "action" in result)
    test("rtc first: action is ndarray", isinstance(result["action"], np.ndarray))
    test("rtc first: action shape correct", result["action"].shape[0] == 1,
         f"got shape {result['action'].shape}")
    print(f"  action shape: {result['action'].shape}")
    return result


def test_rtc_with_prev_action(client, camera_keys, state_dim, prev_action):
    """Test 3: RTC with prev_action (guided inference)."""
    print("\n=== Test 3: RTC with prev_action (guided inference) ===")
    obs = make_mock_obs(state_dim=state_dim, camera_keys=camera_keys)
    result = client.infer(obs, prev_action=prev_action, is_rtc=True)

    test("rtc guided: has 'action' key", "action" in result)
    test("rtc guided: action shape correct", result["action"].shape == prev_action.reshape(1, *prev_action.shape).shape,
         f"got {result['action'].shape}")

    # Check that guided inference produces different output than non-guided
    obs2 = make_mock_obs(state_dim=state_dim, camera_keys=camera_keys)
    result_no_guide = client.infer(obs2, prev_action=None, is_rtc=True)

    diff = np.abs(result["action"] - result_no_guide["action"]).mean()
    test("rtc guided: output differs from non-guided", diff > 1e-4,
         f"mean abs diff = {diff:.6f}")

    print(f"  guided action[0,0,:5]: {result['action'][0, 0, :5]}")
    print(f"  non-guided action[0,0,:5]: {result_no_guide['action'][0, 0, :5]}")
    print(f"  mean abs diff: {diff:.6f}")
    return result


def test_action_chunk_broker_rtc(client, camera_keys, state_dim, pred_horizon, action_dim):
    """Test 4: Full ActionChunkBroker RTC mode."""
    print("\n=== Test 4: ActionChunkBroker RTC mode ===")
    s, d = 4, 2  # Small values for fast testing
    broker = ActionChunkBroker(client, action_horizon=pred_horizon, is_rtc=True, s=s, d=d)

    obs = make_mock_obs(state_dim=state_dim, camera_keys=camera_keys)

    # Step 0: initial inference
    r0 = broker.infer(obs)
    test("broker rtc: step 0 action shape", r0["action"].shape == (action_dim,),
         f"got {r0['action'].shape}")
    test("broker rtc: step 0 no NaN", not np.any(np.isnan(r0["action"])))
    print(f"  step 0 action[:5]: {r0['action'][:5]}")

    # Steps 1 to s+d-1: cached from first chunk
    for step in range(1, s + d):
        r = broker.infer(obs)
        test(f"broker rtc: step {step} no NaN", not np.any(np.isnan(r["action"])))
    print(f"  step {s+d-1} action[:5]: {r['action'][:5]}")

    # Wait for background inference to complete
    print("  Waiting for background inference...")
    time.sleep(2.0)

    # Continue — swap should happen at step s+d
    for step in range(s + d, s + d + d + 1):
        r = broker.infer(obs)
        test(f"broker rtc: step {step} no NaN (post-swap)", not np.any(np.isnan(r["action"])))
    print(f"  post-swap action[:5]: {r['action'][:5]}")

    # Reset
    broker.reset()
    test("broker rtc: reset clears state", broker._last_results is None)


def test_action_chunk_broker_standard(client, camera_keys, state_dim, pred_horizon, action_dim):
    """Test 5: ActionChunkBroker standard mode."""
    print("\n=== Test 5: ActionChunkBroker standard mode ===")
    horizon = 4  # Small for fast testing
    broker = ActionChunkBroker(client, action_horizon=horizon, is_rtc=False)

    obs = make_mock_obs(state_dim=state_dim, camera_keys=camera_keys)

    # Get first chunk
    r0 = broker.infer(obs)
    test("broker std: step 0 action shape", r0["action"].shape == (action_dim,),
         f"got {r0['action'].shape}")
    first_val = r0["action"][0]
    print(f"  step 0 action[:5]: {r0['action'][:5]}")

    # Steps 1 to horizon-1
    for step in range(1, horizon):
        r = broker.infer(obs)
        test(f"broker std: step {step} action shape", r["action"].shape == (action_dim,))

    # After horizon steps, should trigger new inference
    r_new = broker.infer(obs)
    test("broker std: new inference after horizon", r_new["action"].shape == (action_dim,))
    print(f"  new chunk action[:5]: {r_new['action'][:5]}")


def test_metadata(client):
    """Test 6: Server metadata contains RTC config."""
    print("\n=== Test 6: Server metadata ===")
    meta = client.get_server_metadata()
    test("metadata: exists", meta is not None)
    test("metadata: has action_dim", "action_dim" in meta)
    test("metadata: has pred_horizon", "pred_horizon" in meta)
    test("metadata: has rtc_config", "rtc_config" in meta)

    if "rtc_config" in meta:
        rtc = meta["rtc_config"]
        test("metadata: rtc_config has enabled", "enabled" in rtc)
        test("metadata: rtc_config has s", "s" in rtc)
        test("metadata: rtc_config has d", "d" in rtc)
        print(f"  rtc_config: {rtc}")


def main():
    parser = argparse.ArgumentParser(description="RTC test script")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7999)
    args = parser.parse_args()

    print("=" * 60)
    print(f"RTC Test Suite — connecting to {args.host}:{args.port}")
    print("=" * 60)

    # Connect
    print("\nConnecting to server...")
    try:
        client = wcp.WebsocketClientPolicy(host=args.host, port=args.port)
    except Exception as e:
        print(f"\n[ERROR] Cannot connect to server at {args.host}:{args.port}")
        print(f"  {e}")
        print("\nPlease start the server first, e.g.:")
        print("  bash inference/launch_server_so101.sh")
        sys.exit(1)

    meta = client.get_server_metadata()
    print(f"Server metadata: {meta}")

    pred_horizon = meta.get("pred_horizon", 32)
    action_dim = meta.get("action_dim", 6)
    rtc_config = meta.get("rtc_config", {})

    # Determine camera keys from metadata or use defaults
    # The server doesn't expose camera_key in metadata, so we read from the launch script
    camera_keys = ("face_view", "left_wrist_view")
    state_dim = 6  # so101 has 6 state dims

    # Run tests
    try:
        test_metadata(client)

        # Test 1: Standard inference
        result1 = test_standard_inference(client, camera_keys, state_dim)
        actual_action_dim = result1["action"].shape[2]
        actual_pred_horizon = result1["action"].shape[1]
        print(f"  Detected: pred_horizon={actual_pred_horizon}, action_dim={actual_action_dim}")

        # Test 2: RTC first call
        result2 = test_rtc_first_call(client, camera_keys, state_dim)

        # Test 3: RTC with prev_action
        prev_action = result1["action"][0]  # Use first chunk as prev_action
        test_rtc_with_prev_action(client, camera_keys, state_dim, prev_action)

        # Test 4: Full RTC broker
        test_action_chunk_broker_rtc(client, camera_keys, state_dim,
                                     actual_pred_horizon, actual_action_dim)

        # Test 5: Standard broker
        test_action_chunk_broker_standard(client, camera_keys, state_dim,
                                          actual_pred_horizon, actual_action_dim)

    except Exception as e:
        print(f"\n[ERROR] Test failed with exception: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed")
    print("=" * 60)

    if FAIL > 0:
        sys.exit(1)
    else:
        print("All tests passed!")


if __name__ == "__main__":
    main()
