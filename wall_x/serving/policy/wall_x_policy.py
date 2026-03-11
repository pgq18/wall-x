import logging
from typing import Dict, Any, List
import torch
import numpy as np
import json
from transformers import AutoProcessor

from wall_x.serving.websocket_policy_server import BasePolicy
from wall_x.model.qwen2_5_based.modeling_qwen2_5_vl_act import Qwen2_5_VLMoEForAction
from wall_x.serving.policy.utils import prepare_batch
from wall_x.data.utils import KEY_MAPPINGS

from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class NormStats:
    """Normalization statistics for action and state data."""
    min: torch.Tensor
    max: torch.Tensor
    delta: torch.Tensor


class WallXPolicy(BasePolicy):
    """Policy wrapper for Wall-X model that implements the BasePolicy interface."""

    # Action dimension is fixed at 20 for the model
    ACTION_DIM = 20

    def __init__(
        self,
        model_path: str,
        train_config: dict,
        action_tokenizer_path: str,
        action_dim: int,
        agent_pos_dim: int,
        pred_horizon: int,
        camera_key: List[str],
        device: str = "cuda",
        dtype: str = "bfloat16",
        predict_mode: str = "fast",
        default_prompt: str | None = None,
        min_pixels: int = 4 * 28 * 28,
        max_pixels: int = 16384 * 28 * 28,
        image_factor: int = 28,
        max_length: int = 768,
        fake: bool = False,
        norm_stats_path: str = "data/norm_stats.json",
        dataset_name: str = "lerobot/libero_goal_image",
    ):
        """Initialize the Wall-X policy.

        Args:
            model_path: Path to the pretrained model checkpoint
            action_tokenizer_path: Path to the action tokenizer
            action_dim: Dimension of action space
            pred_horizon: Prediction horizon for actions
            device: Device to run model on ('cuda' or 'cpu')
            dtype: Data type for model ('bfloat16', 'float16', or 'float32')
            predict_mode: Prediction mode ('fast' or 'slow')
            default_prompt: Default text prompt for the model
            min_pixels: Minimum pixels for image resizing
            max_pixels: Maximum pixels for image resizing
            image_factor: Factor for smart resize
            max_length: Maximum sequence length for text
        """
        logger.info(f"Loading Wall-X model from {model_path}")
        self.fake_inference = fake
        # Load normalization statistics
        self.norm_stats = self._load_norm_stats(train_config["norm_stats_path"], train_config["data"]["lerobot_config"]["repo_id"])

        if not self.fake_inference:
            self.model = Qwen2_5_VLMoEForAction.from_pretrained(
                model_path,
                train_config=train_config,
                action_tokenizer_path=action_tokenizer_path,
            )
            self.model.eval()
            self.model = self.model.to(device)

            self.model = self.model.bfloat16()

        # hard code the action dim to 20 for align to wall-x configuration
        self.fixed_action_dim = 20

        self.action_dim = action_dim
        self.agent_pos_dim = agent_pos_dim
        self.pred_horizon = pred_horizon
        self.device = device
        self.predict_mode = predict_mode
        self.default_prompt = default_prompt
        self.camera_key = camera_key

        # Image preprocessing config
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.image_factor = image_factor
        self.max_length = max_length

        # Load processor
        logger.info("Loading processor and tokenizer...")
        self.processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
        self.processor.tokenizer.padding_side = "left"

        # Action buffer for multi-step predictions
        self.action_buffer = []
        self.buffer_index = 0

        logger.info(
            f"Model loaded successfully. Device: {device}, Action dim: {action_dim}, Horizon: {pred_horizon}"
        )

    @property
    def metadata(self) -> Dict[str, Any]:
        """Return metadata about the policy."""
        return {
            "action_dim": self.action_dim,
            "pred_horizon": self.pred_horizon,
            "device": self.device,
            "predict_mode": self.predict_mode,
        }

    def reset(self) -> None:
        """Reset the policy state."""
        self.action_buffer = []
        self.buffer_index = 0
        logger.debug("Policy reset")

    def infer(self, obs: Dict) -> Dict:
        """Infer action from observation.

        Args:
            obs: Dictionary containing:
                - 'image': Image observation (numpy array or PIL Image)
                - 'prompt': Optional text prompt
                - 'state': Optional robot state
                - Other modality-specific observations

        Returns:
            Dictionary containing:
                - 'action': Predicted action (numpy array)
                - Additional metadata
        """
        try:
            # Need to predict new actions
            input_batch = prepare_batch(
                obs,
                self.processor,
                self.camera_key,
                self.agent_pos_dim,
                self.action_dim,
                self.pred_horizon,
                self.fixed_action_dim,
                self.max_length,
                self.image_factor,
                self.min_pixels,
                self.max_pixels,
                self.predict_mode,
                self.device,
            )

            if not self.fake_inference:
                with torch.no_grad():
                    outputs = self.model(
                        **input_batch,
                        action_dim=(
                            self.action_dim
                            if self.predict_mode == "fast"
                            else self.fixed_action_dim
                        ),
                        pred_horizon=self.pred_horizon,
                        mode="predict",
                        predict_mode=self.predict_mode,
                    )

                if outputs["predict_action"] is None:
                    predicted_actions = np.zeros(
                        [1, self.pred_horizon, self.action_dim]
                    ).astype(np.float32)

                predicted_actions = (
                    outputs["predict_action"][:, :, : self.action_dim]
                    .detach()
                    .cpu()
                    .to(torch.float32)
                    .numpy()
                )
            else:
                predicted_actions = self._fake_inference(
                    input_batch, self.pred_horizon
                )

            print(predicted_actions.shape)
            return {"action": predicted_actions}

        except Exception as e:
            logger.error(f"Error during inference: {e}")
            raise

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
        actions = self._normalize(actions, self.norm_stats["action"].min, self.norm_stats["action"].delta)
        return actions
    
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
