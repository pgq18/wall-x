#!/usr/bin/env python3
"""
FPGA VLA Client

作为 TCP client 连接 FPGA server（8_test_demo.cpp），完成 diffusion 模式中
服务器端的计算（action_preprocessor.step / action_proj_back / odeint），
并通过 socket 与 FPGA 协调 10 次 flow matching 迭代。

使用方式：
    client = FPGAVLAClient(fpga_host="127.0.0.1", fpga_port=8001)
    client.setup()
    action = client.infer(obs)
    client.cleanup()
"""

import argparse
import json
import logging
import os
import socket
from typing import Dict, Optional

import numpy as np
import torch
import yaml
from transformers import AutoProcessor

from wall_x.serving.policy.utils import prepare_batch
from wall_x.serving.policy.wall_x_policy import WallXPolicy

logger = logging.getLogger(__name__)

# FPGA 协议常量
VIT_RUN_TOKEN = 504
LLM_HIDDEN_DIM = 2048
ACTION_TOKENS = 32

# 默认 tensor 长度（匹配 8_test_demo.cpp 测试数据）
DEFAULT_TEXT0_TOKENS = 22
DEFAULT_PROPRIO_TOKENS = 7
DEFAULT_TEXT1_TOKENS = 31

# 模型配置
MODEL_PATH = "/home/xieqijia/Project/WALL-OSS/wall-x/workspace/libero/workspace/finetuned_new"
ACTION_TOKENIZER_PATH = "/home/xieqijia/Models/fast"
CONFIG_PATH = "/home/xieqijia/Project/WALL-OSS/wall-x/workspace/libero/config_qact.yml"
REPLAN_STEPS = 32


def encode_command(cmd: str, attrs_bytes: bytes = b"") -> bytes:
    return f"command{cmd}attrs{len(attrs_bytes):04d}".encode("utf-8") + attrs_bytes


def recv_all(sock: socket.socket, n: int) -> bytes:
    data = b""
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise ConnectionError("Socket closed unexpectedly")
        data += chunk
    return data


class FPGATransformerClient:
    """负责通过 socket 与 FPGA server 通信，发送 mode1/mode2 并接收 hidden states。"""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8001,
        text0_tokens: int = DEFAULT_TEXT0_TOKENS,
        proprio_tokens: int = DEFAULT_PROPRIO_TOKENS,
        text1_tokens: int = DEFAULT_TEXT1_TOKENS,
        dump_dir: str = "/tmp/fpga_dump",
    ):
        self.host = host
        self.port = port
        self.sock: Optional[socket.socket] = None
        self.text0_tokens = text0_tokens
        self.proprio_tokens = proprio_tokens
        self.text1_tokens = text1_tokens
        self.dump_dir = dump_dir
        os.makedirs(dump_dir, exist_ok=True)
        self._dump_idx = 0

    def connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.host, self.port))
        logger.info(f"Connected to FPGA server at {self.host}:{self.port}")

    def close(self):
        if self.sock:
            try:
                self.sock.sendall(encode_command("quit", b""))
            except Exception:
                pass
            self.sock.close()
            self.sock = None
            logger.info("FPGA connection closed")

    def _send_mode1(
        self,
        image: np.ndarray,
        text_0: np.ndarray,
        proprio: np.ndarray,
        text_1: np.ndarray,
        action: np.ndarray,
    ) -> np.ndarray:
        attrs = np.array(
            [
                VIT_RUN_TOKEN,
                1184,
                self.text0_tokens,
                self.proprio_tokens,
                self.text1_tokens,
                ACTION_TOKENS,
                LLM_HIDDEN_DIM,
            ],
            dtype=np.uint16,
        )
        self.sock.sendall(encode_command("vla1", attrs.tobytes()))
        for tensor in [image, text_0, proprio, text_1, action]:
            self.sock.sendall(tensor.astype(np.uint16).tobytes())
        result = recv_all(self.sock, ACTION_TOKENS * LLM_HIDDEN_DIM * 2)
        result_arr = np.frombuffer(result, dtype=np.uint16).reshape(ACTION_TOKENS, LLM_HIDDEN_DIM)

        # dump
        d = os.path.join(self.dump_dir, f"mode1_{self._dump_idx:04d}")
        os.makedirs(d, exist_ok=True)
        image.tofile(os.path.join(d, "image.bin"))
        text_0.tofile(os.path.join(d, "text_0.bin"))
        proprio.tofile(os.path.join(d, "proprio.bin"))
        text_1.tofile(os.path.join(d, "text_1.bin"))
        action.tofile(os.path.join(d, "action.bin"))
        result_arr.tofile(os.path.join(d, "result.bin"))
        self._dump_idx += 1
        return result_arr

    def _send_mode2(self, action: np.ndarray) -> np.ndarray:
        attrs = np.array([ACTION_TOKENS, LLM_HIDDEN_DIM], dtype=np.uint16)
        self.sock.sendall(encode_command("vla2", attrs.tobytes()))
        self.sock.sendall(action.astype(np.uint16).tobytes())
        result = recv_all(self.sock, ACTION_TOKENS * LLM_HIDDEN_DIM * 2)
        result_arr = np.frombuffer(result, dtype=np.uint16).reshape(ACTION_TOKENS, LLM_HIDDEN_DIM)

        # dump
        d = os.path.join(self.dump_dir, f"mode2_{self._dump_idx:04d}")
        os.makedirs(d, exist_ok=True)
        action.tofile(os.path.join(d, "action.bin"))
        result_arr.tofile(os.path.join(d, "result.bin"))
        self._dump_idx += 1
        return result_arr

    def run_transformer_on_fpga(
        self,
        temp_inputs_embeds: torch.Tensor,
        input_ids: torch.LongTensor,
        action_mask: torch.BoolTensor,
        iteration_idx: int = 0,
    ) -> torch.Tensor:
        """
        将当前 inputs_embeds 拆分为 FPGA 期望的格式，发送 mode1/mode2，
        并返回 FPGA 输出的 hidden_states [ACTION_TOKENS, LLM_HIDDEN_DIM]。
        """
        device = temp_inputs_embeds.device
        dtype = temp_inputs_embeds.dtype
        batch_size, seq_len, hidden_dim = temp_inputs_embeds.shape

        # 找到各特殊 token 的位置
        image_mask = input_ids == 151655  # image_token_id
        propri_mask = input_ids == 151665  # propri_token_id
        # action_mask 已由调用方提供

        image_indices = torch.where(image_mask[0])[0]
        propri_indices = torch.where(propri_mask[0])[0]
        action_indices = torch.where(action_mask[0])[0]

        assert len(image_indices) > 0, "No image tokens found"
        assert len(propri_indices) > 0, "No proprioception tokens found"
        assert len(action_indices) > 0, "No action tokens found"

        img_start = int(image_indices[0].item())
        img_end = int(image_indices[-1].item()) + 1
        propri_start = int(propri_indices[0].item())
        action_start = int(action_indices[0].item())

        # 拆分 text / image / proprio / text1 / action
        text_0 = temp_inputs_embeds[0, :img_start, :]
        image_embeds = temp_inputs_embeds[0, img_start:img_end, :]
        proprio_embeds = temp_inputs_embeds[0, propri_start:propri_start + self.proprio_tokens, :]
        text_1 = temp_inputs_embeds[0, img_end:action_start, :]
        action_embeds = temp_inputs_embeds[0, action_start:action_start + ACTION_TOKENS, :]

        # 对齐到 FPGA 期望的固定长度
        def pad_or_truncate(t, target_len, name):
            cur_len = t.shape[0]
            if cur_len != target_len:
                logger.warning(
                    f"{name} length mismatch: actual={cur_len}, expected={target_len}, "
                    f"will {'truncate' if cur_len > target_len else 'pad'}."
                )
                if cur_len > target_len:
                    t = t[:target_len]
                else:
                    pad = torch.zeros(target_len - cur_len, hidden_dim, device=device, dtype=dtype)
                    t = torch.cat([t, pad], dim=0)
            return t

        text_0 = pad_or_truncate(text_0, self.text0_tokens, "text_0")
        text_1 = pad_or_truncate(text_1, self.text1_tokens, "text_1")

        # image: FPGA 需要 [648, 1184] 的原始 patch 数据
        # 但 temp_inputs_embeds 中的 image_embeds 已经是经过 ViT 后的 [162, 2048]
        # 因此这里无法直接从 inputs_embeds 中恢复原始 image
        # 解决方案：在 FPGAVLAClient.infer() 中预先把 pixel_values 处理好并传给 fpga_client
        image_uint16 = getattr(self, "_cached_image", None)
        if image_uint16 is None and iteration_idx == 0:
            raise RuntimeError(
                "No cached image available. Please call set_cached_image() before inference."
            )

        # 转换为 numpy uint16 (假设数据已经是 fp16/bf16 范围)
        def to_uint16_np(t: torch.Tensor) -> np.ndarray:
            return t.detach().cpu().to(torch.float16).numpy().view(np.uint16)

        text_0_np = to_uint16_np(text_0)
        proprio_np = to_uint16_np(proprio_embeds)
        text_1_np = to_uint16_np(text_1)
        action_np = to_uint16_np(action_embeds)

        if iteration_idx == 0:
            result = self._send_mode1(image_uint16, text_0_np, proprio_np, text_1_np, action_np)
        else:
            result = self._send_mode2(action_np)

        # FPGA returns fp16 bit patterns as uint16; reinterpret as fp16 before converting to torch
        result_fp16 = np.copy(result).view(np.float16)
        return torch.from_numpy(result_fp16).to(device=device, dtype=torch.bfloat16)

    def set_cached_image(self, pixel_values: Optional[torch.Tensor]):
        """
        预处理 pixel_values 为 FPGA 期望的 [648, 1184] uint16 格式。
        pixel_values 预期为 [648, 1176] 的 float32/float16 张量。
        """
        if pixel_values is None:
            self._cached_image = None
            return

        # 确保形状为 [648, 1176]
        pv = pixel_values.view(VIT_RUN_TOKEN, 1176).cpu().to(torch.float16).numpy()
        # zero-padding 到 [648, 1184]
        pv = np.pad(pv, ((0, 0), (0, 8)), mode="constant")
        self._cached_image = pv.view(np.uint16)


class DataProcessor:
    """数据预处理器，用于归一化状态和动作。"""

    def __init__(self, norm_stats):
        self.norm_stats = norm_stats
        self.action_min_stat = norm_stats["action"].min.cpu().numpy()
        self.action_delta = norm_stats["action"].delta.cpu().numpy()
        self.state_min_stat = norm_stats["state"].min.cpu().numpy()
        self.state_delta = norm_stats["state"].delta.cpu().numpy()

    @classmethod
    def _normalize(cls, action, min_stat, delta):
        """使用 min-max 归一化。"""
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        delta = np.where(delta == 0, np.ones_like(delta), delta)
        x = (action - min_stat) / delta
        x = x * 2 - 1
        x = np.clip(x, -1, 1)
        return x

    def in_process(self, input_data):
        """预处理输入数据。"""
        input_data["state"] = self._normalize(
            input_data["state"], self.state_min_stat, self.state_delta
        )
        return input_data


class FPGAVLAClient:
    """高层次的 VLA 推理客户端，封装了 WallXPolicy + FPGA transformer 调用。"""

    def __init__(
        self,
        fpga_host: str = "127.0.0.1",
        fpga_port: int = 8001,
        device: str = "cuda",
        predict_mode: str = "diffusion",
    ):
        self.fpga_host = fpga_host
        self.fpga_port = fpga_port
        self.device = device
        self.predict_mode = predict_mode
        self.policy: Optional[WallXPolicy] = None
        self.fpga_client: Optional[FPGATransformerClient] = None
        self.data_processor: Optional[DataProcessor] = None

    def setup(self):
        # Lazy imports to avoid requiring lerobot in environments that only need the client
        from wall_x.data.load_lerobot_dataset import get_data_configs
        from wall_x.data.utils import load_norm_stats

        logger.info("Loading WallXPolicy...")
        config = self._load_config()
        self.policy = WallXPolicy(
            model_path=MODEL_PATH,
            train_config=config,
            action_tokenizer_path=ACTION_TOKENIZER_PATH,
            action_dim=7,
            agent_pos_dim=8,
            pred_horizon=REPLAN_STEPS,
            device=self.device,
            dtype="bfloat16",
            predict_mode=self.predict_mode,
            camera_key=["front_view", "left_wrist_view"],
        )

        # 加载归一化统计
        dataload_config = get_data_configs(config["data"])
        lerobot_config = dataload_config.get("lerobot_config", {})
        norm_stats = load_norm_stats(
            config.get("norm_stats_path", None),
            lerobot_config.get("repo_id", None)
        )
        self.data_processor = DataProcessor(norm_stats)

        self.fpga_client = FPGATransformerClient(
            host=self.fpga_host,
            port=self.fpga_port,
        )
        self.fpga_client.connect()
        logger.info("FPGA VLA Client setup complete.")

    def _load_config(self):
        with open(CONFIG_PATH, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        config["data"]["model_type"] = config.get("model_type")
        return config

    def infer(self, obs: Dict) -> np.ndarray:
        if self.policy is None or self.fpga_client is None:
            raise RuntimeError("Client not set up. Call setup() first.")

        # 数据预处理（归一化 state）
        if self.data_processor is not None:
            obs = self.data_processor.in_process(obs.copy())

        # 准备模型输入
        input_batch = prepare_batch(
            obs,
            self.policy.processor,
            self.policy.camera_key,
            self.policy.agent_pos_dim,
            self.policy.action_dim,
            self.policy.pred_horizon,
            20,  # fixed_action_dim
            self.policy.max_length,
            self.policy.image_factor,
            self.policy.min_pixels,
            self.policy.max_pixels,
            self.policy.predict_mode,
            self.policy.device,
        )

        # 缓存 image 数据到 fpga_client
        pixel_values = input_batch.get("pixel_values")
        self.fpga_client.set_cached_image(pixel_values)

        # 调用模型 predict，启用 use_fpga 分支
        with torch.no_grad():
            outputs = self.policy.model(
                **input_batch,
                action_dim=(
                    self.policy.action_dim
                    if self.predict_mode == "fast"
                    else 20
                ),
                pred_horizon=self.policy.pred_horizon,
                mode="predict",
                predict_mode=self.predict_mode,
                use_fpga=True,
                fpga_client=self.fpga_client,
            )

        if outputs["predict_action"] is None:
            predicted_actions = np.zeros(
                [1, self.policy.pred_horizon, self.policy.action_dim]
            ).astype(np.float32)
        else:
            predicted_actions = (
                outputs["predict_action"][:, :, : self.policy.action_dim]
                .detach()
                .cpu()
                .to(torch.float32)
                .numpy()
            )

        return predicted_actions

    def reset(self):
        if self.policy is not None:
            self.policy.reset()

    def cleanup(self):
        if self.fpga_client is not None:
            self.fpga_client.close()
            self.fpga_client = None


def main():
    parser = argparse.ArgumentParser(description="FPGA VLA Client test")
    parser.add_argument("--fpga-host", type=str, default="127.0.0.1")
    parser.add_argument("--fpga-port", type=int, default=8001)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    client = FPGAVLAClient(fpga_host=args.fpga_host, fpga_port=args.fpga_port)
    client.setup()

    # Dummy observation for connectivity test
    obs = {
        "front_view": np.zeros((256, 256, 3), dtype=np.uint8),
        "left_wrist_view": np.zeros((256, 256, 3), dtype=np.uint8),
        "state": np.zeros(8, dtype=np.float32),
        "prompt": "pick up the cube",
        "dataset_names": ["lerobot/libero_goal_image"],
    }

    try:
        action = client.infer(obs)
        print(f"Predicted action shape: {action.shape}")
    finally:
        client.cleanup()


if __name__ == "__main__":
    main()