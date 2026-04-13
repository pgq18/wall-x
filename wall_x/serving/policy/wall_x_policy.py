import logging
from typing import Dict, Any, List
import torch
import numpy as np
from transformers import AutoProcessor

from wall_x.serving.websocket_policy_server import BasePolicy
from wall_x.model.qwen2_5_based.modeling_qwen2_5_vl_act import Qwen2_5_VLMoEForAction
from wall_x.serving.policy.utils import prepare_batch

logger = logging.getLogger(__name__)


class WallXPolicy(BasePolicy):
    """Policy wrapper for Wall-X model that implements the BasePolicy interface."""

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
        input_image_resolution: int | None = None,
        rtc_s: int = 16,
        rtc_d: int = 8,
        rtc_beta: float = 8.0,
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
        self.input_image_resolution = input_image_resolution

        # Load processor
        logger.info("Loading processor and tokenizer...")
        self.processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
        self.processor.tokenizer.padding_side = "left"

        # Action buffer for multi-step predictions
        self.action_buffer = []
        self.buffer_index = 0

        # RTC parameters
        self.rtc_s = rtc_s
        self.rtc_d = rtc_d
        self.rtc_beta = rtc_beta

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

    def infer(self, obs: Dict, prev_action=None, is_rtc: bool = False) -> Dict:
        """Infer action from observation.

        Args:
            obs: Dictionary containing:
                - 'image': Image observation (numpy array or PIL Image)
                - 'prompt': Optional text prompt
                - 'state': Optional robot state
                - Other modality-specific observations
            prev_action: Previous action chunk for RTC guided inference (numpy array, unnormalized).
            is_rtc: Whether to use Real-Time Action Chunking mode.

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
                self.input_image_resolution,
            )

            # Prepare prev_action for guided inference
            prev_action_tensor = None
            if is_rtc and prev_action is not None:
                prev_action_tensor = torch.tensor(
                    prev_action, device=self.device, dtype=torch.float32
                )
                if prev_action_tensor.ndim == 2:
                    prev_action_tensor = prev_action_tensor.unsqueeze(0)

                dataset_names = input_batch.get("dataset_names", ["default"])
                if isinstance(dataset_names, str):
                    dataset_names = [dataset_names]
                ds_name = dataset_names[0]

                # Normalize only real action dims manually (avoids pad dims with delta=0 → NaN)
                normalizer = self.model.action_preprocessor.normalizer_action
                real_dim = min(prev_action_tensor.shape[-1], self.action_dim)
                action_min = normalizer.min[ds_name][:real_dim].to(self.device)
                action_delta = normalizer.delta[ds_name][:real_dim].to(self.device)
                # (x - min) / delta, then scale to [-1, 1]
                x_real = (prev_action_tensor[:, :, :real_dim] - action_min) / action_delta
                x_real = x_real * 2 - 1
                x_real = torch.clamp(x_real, -1, 1)

                # Pad to fixed_action_dim with zeros
                if real_dim < self.fixed_action_dim:
                    pad = torch.zeros(
                        prev_action_tensor.shape[0],
                        prev_action_tensor.shape[1],
                        self.fixed_action_dim - real_dim,
                        device=self.device, dtype=prev_action_tensor.dtype,
                    )
                    prev_action_tensor = torch.cat([x_real, pad], dim=-1)
                else:
                    prev_action_tensor = x_real

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
                    prev_action=prev_action_tensor,
                    rtc_s=self.rtc_s,
                    rtc_d=self.rtc_d,
                    rtc_beta=self.rtc_beta,
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

            return {"action": predicted_actions}

        except Exception as e:
            logger.error(f"Error during inference: {e}")
            raise
