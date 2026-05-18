import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "packages"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from fpga_vla_client import FPGATransformerClient
import yaml
import torch
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from wall_x.model.qwen2_5_based.modeling_qwen2_5_vl_act import Qwen2_5_VLMoEForAction
from wall_x.data.load_lerobot_dataset import load_test_dataset, get_data_configs


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config["data"]["model_type"] = config.get("model_type")

    return config


def interpolate_chunk_boundaries(traj, pred_horizon, interp_steps):
    """Linearly interpolate at chunk boundaries for continuity."""
    if interp_steps <= 0:
        return traj.clone()

    smoothed = traj.clone()
    num_chunks = traj.shape[0] // pred_horizon

    for chunk_idx in range(1, num_chunks):
        boundary_idx = chunk_idx * pred_horizon
        prev_last = smoothed[boundary_idx - 1]

        for i in range(min(interp_steps, pred_horizon)):
            alpha = (i + 1) / interp_steps
            smoothed[boundary_idx + i] = (
                (1 - alpha) * prev_last + alpha * traj[boundary_idx + i]
            )

    return smoothed


def mean_filter_trajectory(traj, window_size):
    """Apply a centered sliding-window mean filter."""
    if window_size <= 1:
        return traj.clone()

    if window_size % 2 == 0:
        window_size += 1

    half_w = window_size // 2
    total = traj.shape[0]
    filtered = torch.zeros_like(traj)

    for i in range(total):
        start = max(0, i - half_w)
        end = min(total, i + half_w + 1)
        filtered[i] = traj[start:end].mean(dim=0)

    return filtered


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_horizon", type=int, default=32)
    parser.add_argument("--origin_action_dim", type=int, default=7)
    parser.add_argument("--fpga-host", type=str, default="192.168.50.40")
    parser.add_argument("--fpga-port", type=int, default=8001)
    parser.add_argument("--interp-steps", type=int, default=8,
                        help="Interpolation steps at chunk boundaries (0=disable)")
    parser.add_argument("--filter-window", type=int, default=5,
                        help="Mean filter window size (1=disable)")
    args = parser.parse_args()

    origin_action_dim = args.origin_action_dim
    pred_horizon = args.pred_horizon
    interp_steps = args.interp_steps
    filter_window = args.filter_window

    # get train config
    model_path = "/root/Models/libero_goal_finetuned_new"
    action_tokenizer_path = "/path/to/action/tokenizer"
    save_dir = "/userdata/root/Workspace/wall-x/workspace/libero"
    path = "/userdata/root/Workspace/wall-x/workspace/libero/config_qact_rk3588.yml"
    config = load_config(path)

    # load model with customized robot config
    model = Qwen2_5_VLMoEForAction.from_pretrained(
        model_path, train_config=config, action_tokenizer_path=action_tokenizer_path, skip_transformer_weights=True
    )
    model.eval()
    model = model.to("cpu")
    model = model.float()

    # connect to FPGA
    print("Connecting to FPGA...")
    fpga_client = FPGATransformerClient(host=args.fpga_host, port=args.fpga_port)
    fpga_client.connect()

    # get test dataloader
    dataload_config = get_data_configs(config["data"])
    lerobot_config = dataload_config.get("lerobot_config", {})
    dataset = load_test_dataset(config, lerobot_config, seed=42)
    dataloader = dataset.get_dataloader()

    total_frames = len(dataloader)

    predict_mode = "fast" if config.get("use_fast_tokenizer", False) else "diffusion"
    action_dim = 20 if predict_mode == "diffusion" else origin_action_dim
    gt_traj = torch.zeros((total_frames, origin_action_dim))
    pred_traj = torch.zeros((total_frames, origin_action_dim))

    # use tqdm to show the progress
    for idx, batch in tqdm(
        enumerate(dataloader), total=total_frames, desc="predicting"
    ):
        if idx % pred_horizon == 0 and idx + pred_horizon < total_frames:
            batch = batch.to("cpu")

            # cache image for FPGA mode1
            pixel_values = batch.get("pixel_values")
            if pixel_values is not None:
                fpga_client.set_cached_image(pixel_values)

            with torch.no_grad():
                outputs = model(
                    **batch,
                    action_dim=action_dim,
                    pred_horizon=pred_horizon,
                    mode="predict",
                    predict_mode=predict_mode,
                    fpga_client=fpga_client,
                )
                pred_traj[idx : idx + pred_horizon] = (
                    outputs["predict_action"][:, :, :origin_action_dim]
                    .detach()
                    .cpu()
                    .squeeze(0)
                )

            # Denormalize ground truth actions
            gt_action_chunk = batch["action_chunk"][:, :, :origin_action_dim]
            dof_mask = batch["dof_mask"].to(gt_action_chunk.dtype)
            denormalized_gt = (
                model.action_preprocessor.normalizer_action.unnormalize_data(
                    gt_action_chunk,
                    [lerobot_config.get("repo_id", "physical-intelligence/libero")],
                    dof_mask,
                ).squeeze(0)
            )
            gt_traj[idx : idx + pred_horizon] = denormalized_gt.detach().cpu()

    gt_traj_np = gt_traj.numpy()
    pred_traj_np = pred_traj.numpy()
    smooth_traj = interpolate_chunk_boundaries(pred_traj, pred_horizon, interp_steps)
    smooth_traj = mean_filter_trajectory(smooth_traj, filter_window)
    smooth_traj_np = smooth_traj.numpy()

    timesteps = gt_traj.shape[0]

    fig, axs = plt.subplots(
        origin_action_dim, 1, figsize=(15, 5 * origin_action_dim), sharex=True
    )
    fig.suptitle("Action Comparison for lerobot", fontsize=16)

    for i in range(origin_action_dim):
        axs[i].plot(range(timesteps), gt_traj_np[:, i], label="Ground Truth")
        axs[i].plot(range(timesteps), pred_traj_np[:, i], label="Prediction")
        axs[i].plot(range(timesteps), smooth_traj_np[:, i],
                    label="Smoothed Prediction", color="green", linestyle="--", alpha=0.8)
        axs[i].set_ylabel(f"Action Dim {i+1}")
        axs[i].legend()
        axs[i].grid(True)

    axs[-1].set_xlabel("Timestep")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(save_dir, exist_ok=True)
    suffix = "_smooth" if interp_steps > 0 or filter_window > 1 else ""
    save_path = os.path.join(save_dir, f"lerobot_comparison{suffix}.png")
    plt.savefig(save_path)
    print(f"Saved plot to {save_path}")
    plt.close()
