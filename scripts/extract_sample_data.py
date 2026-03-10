"""
Extract sample data from LeRobot dataset for testing.

This script extracts images, instructions, proprioception, and actions
from the dataset specified in the config file and saves them to disk
for testing the inference pipeline.

Usage:
    # Extract samples only
    python scripts/extract_sample_data.py \
        --config workspace/libero/config_qact.yml \
        --output_dir extracted_samples \
        --num_samples 5 \
        --episode 0

    # Extract and test with fake inference
    python scripts/extract_sample_data.py \
        --config workspace/libero/config_qact.yml \
        --output_dir extracted_samples \
        --test \
        --norm_stats workspace/libero/lerobot/libero_goal_image/norm_stats.json

    # Extract and test with real model inference
    python scripts/extract_sample_data.py \
        --config workspace/libero/config_qact.yml \
        --output_dir extracted_samples \
        --test \
        --no_fake \
        --model_path /path/to/model \
        --norm_stats workspace/libero/lerobot/libero_goal_image/norm_stats.json
"""

import os
import json
import yaml
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
from PIL import Image
from typing import Dict, Any

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from wall_x.data.load_lerobot_dataset import get_data_configs, KEY_MAPPINGS


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config["data"]["model_type"] = config.get("model_type", "qwen2_5")
    return config


def extract_samples(
    config: Dict[str, Any],
    output_dir: str,
    num_samples: int = 5,
    episode: int = 0,
    root_override: str = None,
):
    """
    Extract sample data from the dataset.

    Args:
        config: Configuration dictionary
        output_dir: Directory to save extracted samples
        num_samples: Number of samples to extract
        episode: Episode index to extract from
        root_override: Override dataset root path
    """
    # Get dataset config
    dataload_config = get_data_configs(config["data"])
    lerobot_config = dataload_config.get("lerobot_config", {})

    repo_id = lerobot_config.get("repo_id")
    root = root_override if root_override else lerobot_config.get("root")

    print(f"Loading dataset: {repo_id}")
    print(f"Root: {root}")
    print(f"Episode: {episode}")

    # Get dataset metadata
    meta_info = LeRobotDatasetMetadata(repo_id, root=root)
    dataset_fps = meta_info.fps

    # Get key mappings
    key_mapping = KEY_MAPPINGS.get(repo_id)
    if key_mapping is None:
        raise ValueError(f"No key mapping found for dataset: {repo_id}")

    state_key = key_mapping["state"]
    action_key = key_mapping["action"]
    camera_mapping = key_mapping["camera"]

    print(f"State key: {state_key}")
    print(f"Action key: {action_key}")
    print(f"Camera mapping: {camera_mapping}")

    # Setup delta timestamps for action chunk
    action_horizon = dataload_config.get("action_horizon", 33) - 1
    delta_timestamps = {
        action_key: [t / dataset_fps for t in range(action_horizon)],
    }

    # Load dataset
    dataset = LeRobotDataset(
        repo_id,
        episodes=[episode],
        delta_timestamps=delta_timestamps,
        video_backend="pyav",
        root=root,
    )

    print(f"Dataset loaded: {dataset.num_frames} frames")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get camera keys from dataset metadata
    camera_keys = dataset.meta.camera_keys
    print(f"Camera keys: {camera_keys}")

    # Limit samples to available frames
    num_samples = min(num_samples, dataset.num_frames)
    print(f"Extracting {num_samples} samples...")

    # Metadata storage
    metadata = {
        "repo_id": repo_id,
        "episode": episode,
        "num_samples": num_samples,
        "state_key": state_key,
        "action_key": action_key,
        "camera_keys": camera_keys,
        "camera_mapping": camera_mapping,
        "action_horizon": action_horizon,
        "samples": [],
    }

    # Extract samples
    for i in range(num_samples):
        sample_dir = os.path.join(output_dir, f"sample_{i}")
        os.makedirs(sample_dir, exist_ok=True)

        # Get raw data from dataset
        data = dataset[i]

        sample_info = {
            "index": i,
            "frame_index": data.get("frame_index", i).item() if torch.is_tensor(data.get("frame_index")) else data.get("frame_index", i),
            "episode_index": data.get("episode_index", episode).item() if torch.is_tensor(data.get("episode_index")) else data.get("episode_index", episode),
        }

        print(f"\n--- Sample {i} ---")

        # Save images
        for cam_key in camera_keys:
            if cam_key in data:
                img_tensor = data[cam_key]  # Shape: [C, H, W]

                # Convert to PIL Image
                # Assume values are in [0, 1] range
                if img_tensor.dtype == torch.float32 or img_tensor.dtype == torch.float:
                    img_array = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                else:
                    img_array = img_tensor.permute(1, 2, 0).cpu().numpy()

                img = Image.fromarray(img_array)

                # Get friendly camera name
                cam_name = camera_mapping.get(cam_key, cam_key.replace(".", "_"))
                img_path = os.path.join(sample_dir, f"image_{cam_name}.png")
                img.save(img_path)
                print(f"  Saved image: {img_path} (shape: {img.size})")
                sample_info[f"image_{cam_name}"] = f"image_{cam_name}.png"

        # Save instruction
        instruction = data.get("task", "")
        if isinstance(instruction, torch.Tensor):
            # Handle tensor instruction (might be encoded)
            instruction = str(instruction.item()) if instruction.numel() == 1 else str(instruction)

        instruction_path = os.path.join(sample_dir, "instruction.txt")
        with open(instruction_path, "w") as f:
            f.write(instruction)
        print(f"  Saved instruction: {instruction[:50]}..." if len(instruction) > 50 else f"  Saved instruction: {instruction}")
        sample_info["instruction"] = instruction

        # Save proprioception (state)
        if state_key in data:
            proprio = data[state_key]
            if isinstance(proprio, torch.Tensor):
                proprio = proprio.cpu().numpy()
            proprio_path = os.path.join(sample_dir, "proprio.npy")
            np.save(proprio_path, proprio)
            print(f"  Saved proprio: shape {proprio.shape}")
            sample_info["proprio_shape"] = list(proprio.shape)

        # Save action (ground truth)
        if action_key in data:
            action = data[action_key]
            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy()
            action_path = os.path.join(sample_dir, "action_gt.npy")
            np.save(action_path, action)
            print(f"  Saved action_gt: shape {action.shape}")
            sample_info["action_shape"] = list(action.shape)

        metadata["samples"].append(sample_info)

    # Save metadata
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nSaved metadata to: {metadata_path}")

    print(f"\n{'='*50}")
    print(f"Extraction complete!")
    print(f"  - Output directory: {output_dir}")
    print(f"  - Samples extracted: {num_samples}")
    print(f"{'='*50}")

    return metadata


def draw_comparison_plot(
    output_dir: str,
    metadata: Dict[str, Any],
):
    """
    Draw comparison plots between predictions and ground truth actions.

    Args:
        output_dir: Directory containing extracted samples
        metadata: Metadata dictionary
    """
    print(f"\n{'='*50}")
    print("Drawing comparison plots...")

    for sample_info in metadata["samples"]:
        sample_idx = sample_info["index"]
        sample_dir = os.path.join(output_dir, f"sample_{sample_idx}")

        pred_path = os.path.join(sample_dir, "action_pred.npy")
        gt_path = os.path.join(sample_dir, "action_gt.npy")

        if not os.path.exists(pred_path) or not os.path.exists(gt_path):
            print(f"  Skipping sample {sample_idx}: prediction or ground truth not found")
            continue

        pred = np.load(pred_path)
        gt = np.load(gt_path)

        # Use the minimum length if they differ
        timesteps = min(pred.shape[0], gt.shape[0])
        action_dim = pred.shape[1]

        fig, axs = plt.subplots(
            action_dim, 1, figsize=(15, 5 * action_dim), sharex=True
        )
        if action_dim == 1:
            axs = [axs]

        instruction = sample_info.get("instruction", "N/A")
        fig.suptitle(
            f"Action Comparison - Sample {sample_idx}\nInstruction: {instruction[:50]}{'...' if len(instruction) > 50 else ''}",
            fontsize=14
        )

        for i in range(action_dim):
            axs[i].plot(range(timesteps), gt[:timesteps, i], label="Ground Truth", linewidth=2)
            axs[i].plot(range(timesteps), pred[:timesteps, i], label="Prediction", linewidth=2, linestyle='--')
            axs[i].set_ylabel(f"Dim {i+1}")
            axs[i].legend(loc="upper right")
            axs[i].grid(True, alpha=0.3)

        axs[-1].set_xlabel("Timestep")
        plt.tight_layout(rect=[0, 0.03, 1, 0.93])

        plot_path = os.path.join(sample_dir, "comparison_plot.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"  Saved plot: {plot_path}")

    print(f"{'='*50}\n")


def test_with_inference_script(
    output_dir: str,
    config_path: str,
    norm_stats_path: str,
    use_fake_inference: bool = True,
    model_path: str = None,
    draw_plot: bool = True,
):
    """
    Test the extracted samples with test_inference.py.

    Args:
        output_dir: Directory containing extracted samples
        config_path: Path to config file
        norm_stats_path: Path to norm_stats.json
        use_fake_inference: Whether to use fake inference
        model_path: Path to model weights (required if use_fake_inference=False)
        draw_plot: Whether to draw comparison plots
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from test_inference import CustomInferenceTester

    # Load metadata
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    print(f"\nTesting with test_inference.py...")
    print(f"  - Config: {config_path}")
    print(f"  - Norm stats: {norm_stats_path}")
    print(f"  - Fake inference: {use_fake_inference}")
    if not use_fake_inference:
        print(f"  - Model path: {model_path}")

    # Determine origin_action_dim from action shape (not proprio shape)
    sample_0_action = np.load(os.path.join(output_dir, "sample_0", "action_gt.npy"))
    origin_action_dim = sample_0_action.shape[-1]

    # Create tester
    tester = CustomInferenceTester(
        config_path=config_path,
        norm_stats_path=norm_stats_path,
        origin_action_dim=origin_action_dim,
        use_fake_inference=use_fake_inference,
        model_path=model_path,
        dataset_name=metadata["repo_id"],
    )

    # Set camera mapping from metadata
    from collections import OrderedDict
    cam_mapping = OrderedDict()
    for cam_key, cam_name in metadata["camera_mapping"].items():
        cam_mapping[cam_key] = cam_name
    tester.set_camera_mapping(cam_mapping)

    # Test each sample
    for sample_info in metadata["samples"]:
        sample_dir = os.path.join(output_dir, f"sample_{sample_info['index']}")

        print(f"\n{'='*50}")
        print(f"Testing sample {sample_info['index']}")

        # Load images
        images = []
        for cam_key in metadata["camera_keys"]:
            cam_name = metadata["camera_mapping"].get(cam_key, cam_key.replace(".", "_"))
            img_path = os.path.join(sample_dir, f"image_{cam_name}.png")
            if os.path.exists(img_path):
                images.append(Image.open(img_path))

        # Load instruction
        with open(os.path.join(sample_dir, "instruction.txt"), "r") as f:
            instruction = f.read()

        # Load proprio
        proprio = np.load(os.path.join(sample_dir, "proprio.npy"))

        # Run prediction
        actions = tester.predict(
            images=images,
            instruction=instruction,
            proprio=proprio,
            pred_horizon=metadata["action_horizon"],
        )

        print(f"  Prediction shape: {actions.shape}")

        # Load ground truth for comparison
        action_gt_path = os.path.join(sample_dir, "action_gt.npy")
        if os.path.exists(action_gt_path):
            action_gt = np.load(action_gt_path)
            print(f"  Ground truth shape: {action_gt.shape}")

            # Save prediction
            pred_path = os.path.join(sample_dir, "action_pred.npy")
            np.save(pred_path, actions)
            print(f"  Saved prediction to: {pred_path}")

    # Draw comparison plots if requested
    if draw_plot:
        draw_comparison_plot(output_dir, metadata)


def main():
    parser = argparse.ArgumentParser(description="Extract sample data from LeRobot dataset")
    parser.add_argument(
        "--config",
        type=str,
        default="workspace/libero/config_qact.yml",
        help="Path to config file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="extracted_samples",
        help="Directory to save extracted samples",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of samples to extract",
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=0,
        help="Episode index to extract from",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Override dataset root path",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test extracted samples with test_inference.py",
    )
    parser.add_argument(
        "--norm_stats",
        type=str,
        default=None,
        help="Path to norm_stats.json (required if --test is set)",
    )
    parser.add_argument(
        "--no_fake",
        action="store_true",
        help="Use real model inference instead of fake (requires model)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to model weights (required if --no_fake is set)",
    )
    parser.add_argument(
        "--no_plot",
        action="store_true",
        help="Disable drawing comparison plots",
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Extract samples
    extract_samples(
        config=config,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        episode=args.episode,
        root_override=args.root,
    )

    # Test with inference script if requested
    if args.test:
        if args.norm_stats is None:
            # Try to get norm_stats_path from config
            args.norm_stats = config.get("norm_stats_path")
            if args.norm_stats is None:
                print("Error: --norm_stats is required when using --test")
                return

        test_with_inference_script(
            output_dir=args.output_dir,
            config_path=args.config,
            norm_stats_path=args.norm_stats,
            use_fake_inference=not args.no_fake,
            model_path=args.model_path,
            draw_plot=not args.no_plot,
        )


if __name__ == "__main__":
    main()
