"""
Test real model inference with extracted samples and draw comparison plots.
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import OrderedDict
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from test_inference import CustomInferenceTester


def load_metadata(output_dir: str):
    """Load metadata from output directory."""
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    return metadata


def test_real_inference(
    output_dir: str,
    config_path: str,
    norm_stats_path: str,
    model_path: str,
):
    """Test real model inference with extracted samples."""
    metadata = load_metadata(output_dir)

    print(f"\n{'='*60}")
    print("Testing with REAL model inference...")
    print(f"  - Config: {config_path}")
    print(f"  - Norm stats: {norm_stats_path}")
    print(f"  - Model: {model_path}")
    print(f"  - Dataset: {metadata['repo_id']}")
    print(f"{'='*60}\n")

    # Determine action dim from ground truth
    sample_0_action = np.load(os.path.join(output_dir, "sample_0", "action_gt.npy"))
    origin_action_dim = sample_0_action.shape[-1]
    print(f"Origin action dim: {origin_action_dim}")

    # Create tester with real model
    print("\nLoading model (this may take a while)...")
    tester = CustomInferenceTester(
        config_path=config_path,
        norm_stats_path=norm_stats_path,
        origin_action_dim=origin_action_dim,
        use_fake_inference=False,
        model_path=model_path,
        dataset_name=metadata["repo_id"],
    )

    # Set camera mapping
    cam_mapping = OrderedDict()
    for cam_key, cam_name in metadata["camera_mapping"].items():
        cam_mapping[cam_key] = cam_name
    tester.set_camera_mapping(cam_mapping)
    print("Model loaded successfully!\n")

    # Test each sample
    for sample_info in metadata["samples"]:
        sample_idx = sample_info["index"]
        sample_dir = os.path.join(output_dir, f"sample_{sample_idx}")

        print(f"\n{'='*50}")
        print(f"Testing sample {sample_idx}")
        print(f"  Instruction: {sample_info['instruction']}")
        print(f"{'='*50}")

        # Load images
        images = []
        for cam_key in metadata["camera_keys"]:
            cam_name = metadata["camera_mapping"].get(cam_key, cam_key.replace(".", "_"))
            img_path = os.path.join(sample_dir, f"image_{cam_name}.png")
            if os.path.exists(img_path):
                images.append(Image.open(img_path))
                print(f"  Loaded image: {img_path}")

        # Load instruction
        with open(os.path.join(sample_dir, "instruction.txt"), "r") as f:
            instruction = f.read()

        # Load proprio
        proprio = np.load(os.path.join(sample_dir, "proprio.npy"))
        print(f"  Proprio shape: {proprio.shape}")

        # Load ground truth action
        action_gt = np.load(os.path.join(sample_dir, "action_gt.npy"))
        print(f"  Ground truth shape: {action_gt.shape}")

        # Run prediction
        print("\n  Running model prediction...")
        actions = tester.predict(
            images=images,
            instruction=instruction,
            proprio=proprio,
            pred_horizon=metadata["action_horizon"],
        )
        print(f"  Prediction shape: {actions.shape}")

        # Save prediction
        pred_path = os.path.join(sample_dir, "action_pred.npy")
        np.save(pred_path, actions)
        print(f"  Saved prediction: {pred_path}")

        # Draw comparison plot
        draw_comparison_plot(sample_dir, sample_idx, sample_info['instruction'], action_gt, actions)

    print(f"\n{'='*60}")
    print("All samples tested!")
    print(f"{'='*60}")


def draw_comparison_plot(sample_dir, sample_idx, instruction, gt, pred):
    """Draw comparison plot between ground truth and prediction."""
    timesteps = min(gt.shape[0], pred.shape[0])
    action_dim = gt.shape[1]

    fig, axs = plt.subplots(
        action_dim, 1, figsize=(15, 5 * action_dim), sharex=True
    )
    if action_dim == 1:
        axs = [axs]

    title = f"Action Comparison - Sample {sample_idx}\nInstruction: {instruction[:50]}{'...' if len(instruction) > 50 else ''}"
    fig.suptitle(title, fontsize=14)

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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test real model inference")
    parser.add_argument("--output_dir", type=str, default="extracted_samples")
    parser.add_argument("--config", type=str, default="workspace/libero/config_qact.yml")
    parser.add_argument("--norm_stats", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)

    args = parser.parse_args()

    test_real_inference(
        output_dir=args.output_dir,
        config_path=args.config,
        norm_stats_path=args.norm_stats,
        model_path=args.model_path,
    )
