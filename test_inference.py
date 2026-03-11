"""
Test inference script for custom inputs without dataset loader.

This script allows testing the data processing pipeline with custom:
- Images (PIL Image or file paths)
- Text instructions
- Robot proprioception state

The model inference can be either:
- Fake mode: Random output for quick testing
- Real mode: Actual model inference

Usage:
    # Fake mode (default, for testing data processing)
    python test_inference.py --config workspace/libero/config_qact.yml

    # Real model inference
    python test_inference.py --config workspace/libero/config_qact.yml --model_path /path/to/model --no-fake
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import json
import yaml
import time
import torch
import numpy as np
import argparse
from PIL import Image
from typing import List, Union, Optional, Dict, Any
from collections import OrderedDict
from dataclasses import dataclass

from transformers import AutoProcessor
from qwen_vl_utils.vision_process import smart_resize
from wall_x.data.utils import (
    preprocesser_call,
    get_wallx_normal_text,
    CAMERA_NAME_MAPPING,
    load_norm_stats,
    KEY_MAPPINGS,
)


@dataclass
class NormStats:
    """Normalization statistics for action and state data."""
    min: torch.Tensor
    max: torch.Tensor
    delta: torch.Tensor


class CustomInferenceTester:
    """
    Custom inference tester for Wall-X robot action prediction.

    This class provides a complete data processing pipeline that accepts
    custom inputs (images, instructions, proprio) and outputs executable
    action sequences for robots.

    Supports two modes:
    - Fake mode: Uses random output for quick testing of data processing
    - Real mode: Uses actual model for inference
    """

    # Action dimension is fixed at 20 for the model
    ACTION_DIM = 20

    def __init__(
        self,
        config_path: str,
        norm_stats_path: str,
        origin_action_dim: int = 7,
        use_fake_inference: bool = True,
        model_path: Optional[str] = None,
        predict_mode: str = "diffusion",
        dataset_name: str = "physical-intelligence/libero",
    ):
        """
        Initialize the custom inference tester.

        Args:
            config_path: Path to the YAML configuration file
            norm_stats_path: Path to the norm_stats.json file
            origin_action_dim: Original action dimension (e.g., 7 for 7-DOF robot)
            use_fake_inference: If True, use random output instead of real model
            model_path: Path to model weights (required if use_fake_inference=False)
            predict_mode: Prediction mode ("diffusion" or "fast")
            dataset_name: Dataset name for normalization stats lookup
        """
        self.config_path = config_path
        self.norm_stats_path = norm_stats_path
        self.origin_action_dim = origin_action_dim
        self.use_fake_inference = use_fake_inference
        self.predict_mode = predict_mode
        self.dataset_name = dataset_name

        # Load configuration
        self.config = self._load_config(config_path)

        # Load processor
        self.processor = self._load_processor()

        # Load normalization statistics
        self.norm_stats = self._load_norm_stats(norm_stats_path, dataset_name)

        # Load model if not using fake inference
        self.model = None
        if not use_fake_inference:
            if model_path is None:
                raise ValueError("model_path is required when use_fake_inference=False")
            self.model = self._load_model(model_path)

        # Default camera mapping (can be customized)
        self.cam_mapping = OrderedDict([
            ("observation.images.image", "face_view"),
            ("observation.images.wrist_image", "left_wrist_view"),
        ])

        print(f"CustomInferenceTester initialized:")
        print(f"  - Origin action dim: {origin_action_dim}")
        print(f"  - Use fake inference: {use_fake_inference}")
        print(f"  - Predict mode: {predict_mode}")
        print(f"  - Dataset name: {dataset_name}")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        config["data"]["model_type"] = config.get("model_type", "qwen2_5")
        return config

    def _load_processor(self) -> AutoProcessor:
        """Load and configure the multimodal processor."""
        processor_path = self.config["pretrained_wallx_path"]
        processor = AutoProcessor.from_pretrained(processor_path, use_fast=True)

        # Set padding side
        if self.config.get("padding_side", "left") == "left":
            processor.tokenizer.padding_side = "left"

        # Add special tokens
        new_tokens = ["<|propri|>", "<|action|>"]
        processor.tokenizer.add_tokens(new_tokens)

        # Add action tokens for fast tokenizer
        if self.config.get("use_fast_tokenizer", False) and self.config.get("model_type") == "qwen2_5":
            action_tokenizer_path = self.config.get("action_tokenizer_path")
            if action_tokenizer_path:
                action_tokenizer = AutoProcessor.from_pretrained(
                    action_tokenizer_path, trust_remote_code=True
                )
                new_tokens = [f"<|action_token_{i}|>" for i in range(action_tokenizer.vocab_size)]
                processor.tokenizer.add_tokens(new_tokens)
                begin_idx_token = "<|action_token_0|>"
                token_id = processor.tokenizer.convert_tokens_to_ids(begin_idx_token)
                processor.tokenizer.init_kwargs["action_token_start_index"] = token_id
                processor.tokenizer.init_kwargs["action_token_vocab_size"] = action_tokenizer.vocab_size

        return processor

    def _load_norm_stats(self, norm_stats_path: str, dataset_name: str) -> Dict[str, NormStats]:
        """Load normalization statistics from JSON file."""
        with open(norm_stats_path, "r") as f:
            norm_stats = json.load(f)

        # Get action stats
        action_key = KEY_MAPPINGS.get(dataset_name, KEY_MAPPINGS["physical-intelligence/libero"])["action"]
        q01 = torch.tensor(norm_stats["norm_stats"][action_key]["q01"])
        q99 = torch.tensor(norm_stats["norm_stats"][action_key]["q99"])
        delta = q99 - q01
        action_norm_stats = NormStats(min=q01, max=q99, delta=delta)

        # Get state stats
        state_key = KEY_MAPPINGS.get(dataset_name, KEY_MAPPINGS["physical-intelligence/libero"])["state"]
        q01 = torch.tensor(norm_stats["norm_stats"][state_key]["q01"])
        q99 = torch.tensor(norm_stats["norm_stats"][state_key]["q99"])
        delta = q99 - q01
        state_norm_stats = NormStats(min=q01, max=q99, delta=delta)

        return {"action": action_norm_stats, "state": state_norm_stats}

    def _load_model(self, model_path: str):
        """Load the real model for inference."""
        from wall_x.model.qwen2_5_based.modeling_qwen2_5_vl_act import Qwen2_5_VLMoEForAction

        model = Qwen2_5_VLMoEForAction.from_pretrained(
            model_path,
            train_config=self.config,
            action_tokenizer_path=self.config.get("action_tokenizer_path"),
        )
        model.eval()
        model = model.to("cuda")
        model = model.bfloat16()
        return model

    def _normalize(
        self,
        data: torch.Tensor,
        min_stat: torch.Tensor,
        delta: torch.Tensor,
    ) -> torch.Tensor:
        """Normalize data to [-1, 1] range using min-max normalization."""
        delta = torch.where(delta == 0, torch.ones_like(delta), delta)
        x = (data - min_stat) / delta
        x = x * 2 - 1
        x = torch.clamp(x, -1, 1)
        return x

    def _unnormalize(
        self,
        data: torch.Tensor,
        min_stat: torch.Tensor,
        delta: torch.Tensor,
    ) -> torch.Tensor:
        """Convert normalized data back to original scale."""
        # Move stats to the same device as data
        min_stat = min_stat.to(data.device)
        delta = delta.to(data.device)
        x = (data + 1) / 2
        x = x * delta + min_stat
        return x

    def _load_images(
        self,
        images: List[Union[Image.Image, str]],
    ) -> List[Image.Image]:
        """Load and convert images to PIL format."""
        loaded_images = []
        for img in images:
            if isinstance(img, str):
                img = Image.open(img).convert("RGB")
            elif not isinstance(img, Image.Image):
                raise ValueError(f"Image must be PIL Image or path, got {type(img)}")
            loaded_images.append(img)
        return loaded_images

    def _resize_images(
        self,
        images: List[Image.Image],
        resolution: Optional[int] = None,
    ) -> List[Image.Image]:
        """Resize images using smart_resize for model compatibility."""
        resolution_config = self.config.get("data", {}).get("resolution", {})
        processed_images = []

        for i, img in enumerate(images):
            # Get target resolution for this camera
            cam_name = list(self.cam_mapping.values())[i] if i < len(self.cam_mapping) else f"cam_{i}"
            target_size = resolution or resolution_config.get(cam_name, 256)

            orig_width, orig_height = img.size

            # Apply resolution constraints
            if orig_width > orig_height:  # Landscape
                new_width = target_size
                new_height = int(target_size * orig_height / orig_width)
            else:  # Portrait
                new_height = target_size
                new_width = int(target_size * orig_width / orig_height)
            img = img.resize((new_width, new_height))

            # Apply smart scaling (qwen logic)
            current_width, current_height = img.size
            resized_height, resized_width = smart_resize(
                current_height,
                current_width,
                factor=28,  # Default image_factor
                min_pixels=3136,  # Default min_pixels
                max_pixels=12845056,  # Default max_pixels
            )
            img = img.resize((resized_width, resized_height))
            processed_images.append(img)

        return processed_images

    def _build_text(
        self,
        instruction: str,
        action_horizon: int = 32,
    ) -> str:
        """Build the formatted prompt text for the model."""
        instruction_info = {"instruction": instruction}
        text, _ = get_wallx_normal_text(
            instruction_info,
            action_chunk_size=action_horizon,
            frame_idx=0,
            priority_order=None,
            cam_mapping=self.cam_mapping,
            generate_subtask_ratio=0.0,
        )
        return text

    def _preprocess_proprio(
        self,
        proprio: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Preprocess proprioception data.

        Args:
            proprio: Raw proprioception data [proprio_dim]

        Returns:
            Tuple of (normalized_proprio, proprio_mask) both [1, 1, 20]
        """
        # Convert to tensor
        proprio = torch.tensor(proprio, dtype=torch.float32)
        proprio_dim = proprio.shape[-1]

        # Handle NaN values
        proprio_mask = (~torch.isnan(proprio)).float()
        proprio = torch.nan_to_num(proprio, nan=0.0)

        # Normalize using state normalization stats
        state_min = self.norm_stats["state"].min[:proprio_dim]
        state_delta = self.norm_stats["state"].delta[:proprio_dim]
        proprio = self._normalize(proprio, state_min, state_delta)

        # Pad to 20 dimensions
        if proprio.shape[-1] < self.ACTION_DIM:
            padding = torch.zeros(self.ACTION_DIM - proprio.shape[-1])
            proprio = torch.cat([proprio, padding], dim=-1)
            proprio_mask = torch.cat([proprio_mask, torch.zeros(self.ACTION_DIM - proprio_mask.shape[-1])], dim=-1)

        # Reshape to [1, 1, 20]
        proprio = proprio.unsqueeze(0).unsqueeze(0)
        proprio_mask = proprio_mask.unsqueeze(0).unsqueeze(0)

        return proprio, proprio_mask

    def _build_batch(
        self,
        images: List[Image.Image],
        instruction: str,
        proprio: np.ndarray,
        pred_horizon: int = 32,
    ) -> Dict[str, torch.Tensor]:
        """
        Build the complete batch for model input.

        Args:
            images: List of PIL images
            instruction: Task instruction text
            proprio: Robot proprioception state [origin_action_dim]
            pred_horizon: Number of actions to predict

        Returns:
            Dictionary containing all required batch fields
        """
        # Resize images
        processed_images = self._resize_images(images)

        # Build text
        text = self._build_text(instruction, action_horizon=pred_horizon)

        # Preprocess proprio
        proprio_tensor, proprio_mask = self._preprocess_proprio(proprio)

        # Create dummy action chunk for batch format (will be ignored in inference)
        # Shape: [1, pred_horizon, 20]
        action_chunk = torch.zeros(1, pred_horizon, self.ACTION_DIM)
        dof_mask = torch.zeros(1, pred_horizon, self.ACTION_DIM)
        dof_mask[:, :, :self.origin_action_dim] = 1.0

        # Process with processor
        inputs = preprocesser_call(
            processor=self.processor,
            text=[text],
            images=[processed_images],
            videos=None,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=768,
        )

        # Get action token ID for MoE token types
        action_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|action|>")

        # Build final batch
        batch = {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "pixel_values": inputs.pixel_values,
            "image_grid_thw": inputs.image_grid_thw,
            "proprioception": proprio_tensor,
            "agent_pos_mask": proprio_mask,
            "action_chunk": action_chunk,
            "dof_mask": dof_mask,
            "moe_token_types": (inputs.input_ids == action_token_id).long(),
            "dataset_names": [self.dataset_name],
        }

        return batch

    def _fake_inference(
        self,
        batch: Dict[str, torch.Tensor],
        pred_horizon: int = 32,
    ) -> torch.Tensor:
        """
        Generate fake random actions for testing.

        Args:
            batch: Input batch (unused in fake mode)
            pred_horizon: Number of actions to predict

        Returns:
            Random action tensor [1, pred_horizon, ACTION_DIM]
        """
        print("[FAKE INFERENCE] Generating random actions...")
        # Generate random actions in [-1, 1] range
        actions = torch.rand(1, pred_horizon, self.ACTION_DIM) * 2 - 1
        return actions

    def _real_inference(
        self,
        batch: Dict[str, torch.Tensor],
        pred_horizon: int = 32,
    ) -> torch.Tensor:
        """
        Run actual model inference.

        Args:
            batch: Input batch
            pred_horizon: Number of actions to predict

        Returns:
            Predicted action tensor [1, pred_horizon, ACTION_DIM]
        """
        print(f"[REAL INFERENCE] Running model prediction (mode={self.predict_mode})...")

        # Move batch to GPU
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to("cuda")
            else:
                device_batch[key] = value

        action_dim = self.ACTION_DIM if self.predict_mode == "diffusion" else self.origin_action_dim

        with torch.no_grad():
            outputs = self.model(
                **device_batch,
                mode="predict",
                predict_mode=self.predict_mode,
                pred_horizon=pred_horizon,
                action_dim=action_dim,
            )

        return outputs["predict_action"]

    def _postprocess_actions(
        self,
        normalized_actions: torch.Tensor,
    ) -> np.ndarray:
        """
        Postprocess actions from normalized to original scale.

        Args:
            normalized_actions: Normalized actions [1, pred_horizon, ACTION_DIM]

        Returns:
            Unnormalized actions [pred_horizon, origin_action_dim]
        """
        # Extract DOF mask for valid dimensions
        dof_mask = torch.ones(
            normalized_actions.shape[1], self.origin_action_dim,
            device=normalized_actions.device
        ).bool()
        dof_mask = dof_mask.unsqueeze(0)  # [1, pred_horizon, origin_action_dim]

        # Unnormalize
        unnormalized = self._unnormalize(
            normalized_actions[:, :, :self.origin_action_dim],
            self.norm_stats["action"].min[:self.origin_action_dim],
            self.norm_stats["action"].delta[:self.origin_action_dim],
        )

        return unnormalized.squeeze(0).cpu().numpy()

    def predict(
        self,
        images: List[Union[Image.Image, str]],
        instruction: str,
        proprio: np.ndarray,
        pred_horizon: int = 32,
    ) -> np.ndarray:
        """
        Predict robot action sequence from custom inputs.

        Args:
            images: List of images (PIL Image or file paths)
            instruction: Task instruction text
            proprio: Robot proprioception state [origin_action_dim]
            pred_horizon: Number of future actions to predict

        Returns:
            Action sequence [pred_horizon, origin_action_dim]
        """
        print(f"\n{'='*50}")
        print(f"Predicting actions...")
        print(f"  - Number of images: {len(images)}")
        print(f"  - Instruction: {instruction}")
        print(f"  - Proprio shape: {proprio.shape}")
        print(f"  - Prediction horizon: {pred_horizon}")
        print(f"{'='*50}\n")

        # Load images
        images = self._load_images(images)

        # Build batch
        batch = self._build_batch(images, instruction, proprio, pred_horizon)

        # Run inference
        if self.use_fake_inference:
            normalized_actions = self._fake_inference(batch, pred_horizon)
        else:
            normalized_actions = self._real_inference(batch, pred_horizon)

        # Postprocess actions
        actions = self._postprocess_actions(normalized_actions)

        print(f"\nPrediction complete!")
        print(f"  - Output shape: {actions.shape}")
        print(f"  - Action range: [{actions.min():.4f}, {actions.max():.4f}]")

        return actions

    def set_camera_mapping(self, cam_mapping: OrderedDict):
        """Set custom camera mapping."""
        self.cam_mapping = cam_mapping


def main():
    parser = argparse.ArgumentParser(description="Test inference with custom inputs")
    parser.add_argument(
        "--config",
        type=str,
        default="workspace/libero/config_qact.yml",
        help="Path to config file",
    )
    parser.add_argument(
        "--norm_stats",
        type=str,
        required=True,
        help="Path to norm_stats.json",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to model weights (required if not using fake inference)",
    )
    parser.add_argument(
        "--fake",
        action="store_true",
        default=True,
        help="Use fake inference (random output)",
    )
    parser.add_argument(
        "--no-fake",
        action="store_false",
        dest="fake",
        help="Use real model inference",
    )
    parser.add_argument(
        "--predict_mode",
        type=str,
        default="diffusion",
        choices=["diffusion", "fast"],
        help="Prediction mode",
    )
    parser.add_argument(
        "--origin_action_dim",
        type=int,
        default=7,
        help="Original action dimension",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="physical-intelligence/libero",
        help="Dataset name for norm stats",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=10,
        help="Number of inference runs for timing measurement",
    )

    args = parser.parse_args()

    # Create tester
    tester = CustomInferenceTester(
        config_path=args.config,
        norm_stats_path=args.norm_stats,
        origin_action_dim=args.origin_action_dim,
        use_fake_inference=args.fake,
        model_path=args.model_path,
        predict_mode=args.predict_mode,
        dataset_name=args.dataset_name,
    )

    # Example usage with dummy data
    print("\n" + "="*60)
    print("Running example prediction with dummy data...")
    print("="*60 + "\n")

    # Create dummy images (you should replace with real images)
    dummy_images = [
        Image.new("RGB", (256, 256), color="red"),
        Image.new("RGB", (256, 256), color="blue"),
    ]

    # Create dummy proprioception (7-DOF robot)
    dummy_proprio = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])

    # Example instruction
    instruction = "Pick up the red block"

    # Run prediction multiple times and measure timing
    latencies = []
    actions = None

    print(f"\n{'='*60}")
    print(f"Running {args.num_runs} inference runs for timing measurement...")
    print(f"{'='*60}\n")

    for i in range(args.num_runs):
        start_time = time.perf_counter()

        actions = tester.predict(
            images=dummy_images,
            instruction=instruction,
            proprio=dummy_proprio,
            pred_horizon=32,
        )

        end_time = time.perf_counter()
        latency = (end_time - start_time) * 1000  # Convert to milliseconds
        latencies.append(latency)
        print(f"Run {i+1}/{args.num_runs}: {latency:.2f} ms")

    # Calculate statistics
    avg_latency = sum(latencies) / len(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)

    print("\n" + "="*60)
    print("Timing Statistics:")
    print("="*60)
    print(f"Number of runs: {args.num_runs}")
    print(f"Average latency: {avg_latency:.2f} ms")
    print(f"Min latency: {min_latency:.2f} ms")
    print(f"Max latency: {max_latency:.2f} ms")
    print(latencies)

    print("\n" + "="*60)
    print("Prediction Results (last run):")
    print("="*60)
    print(f"Action sequence shape: {actions.shape}")
    print(f"First action: {actions[0]}")
    print(f"Last action: {actions[-1]}")
    # print("\nFull action sequence:")
    # print(actions)


if __name__ == "__main__":
    main()
