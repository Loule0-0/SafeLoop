#!/usr/bin/env python
"""Serve the official OpenVLA-OFT LIBERO policy over SafeLoop's websocket protocol."""
from __future__ import annotations

import argparse
import asyncio
import functools
import logging
import os
import random
import sys
import time
import traceback
from dataclasses import dataclass, replace
from pathlib import Path

import msgpack
import numpy as np
import websockets
import websockets.asyncio.server


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OPENVLA_ROOT = PROJECT_ROOT / "third_party" / "openvla-oft"
DEFAULT_CHECKPOINT = "moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10"


@dataclass(frozen=True)
class OpenVLAConfig:
    pretrained_checkpoint: str
    unnorm_key: str
    use_l1_regression: bool = True
    use_diffusion: bool = False
    num_diffusion_steps_train: int = 50
    num_diffusion_steps_inference: int = 50
    use_film: bool = False
    num_images_in_input: int = 2
    use_proprio: bool = True
    center_crop: bool = True
    lora_rank: int = 32
    load_in_8bit: bool = False
    load_in_4bit: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openvla-root", type=Path, default=DEFAULT_OPENVLA_ROOT)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--libero-90-unnorm-key", default="libero_10_no_noops")
    parser.add_argument("--inference-dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    return parser.parse_args()


def configure_imports(openvla_root: Path) -> None:
    paths = [PROJECT_ROOT, openvla_root]
    for path in reversed(paths):
        sys.path.insert(0, str(path.resolve()))


def disable_tensorflow_gpu() -> None:
    """Keep TensorFlow helpers on CPU so PyTorch owns the selected inference GPU."""
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        import tensorflow as tf
        tf.config.set_visible_devices([], "GPU")
    except ImportError:
        return
    except RuntimeError as exc:
        logging.warning("Could not disable TensorFlow GPU visibility: %s", exc)
    finally:
        if cuda_visible_devices is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices


class OpenVLAOFTPolicy:
    def __init__(self, cfg: OpenVLAConfig, libero_90_unnorm_key: str, inference_dtype: str) -> None:
        import torch
        from experiments.robot.openvla_utils import get_action_head, get_processor, get_proprio_projector, get_vla

        self._cfg = cfg
        self._libero_90_unnorm_key = libero_90_unnorm_key
        self._dtype = getattr(torch, inference_dtype)
        self._vla = get_vla(cfg)
        self._processor = get_processor(cfg)
        self._proprio_projector = get_proprio_projector(cfg, self._vla.llm_dim, proprio_dim=8)
        self._action_head = get_action_head(cfg, self._vla.llm_dim)
        if self._dtype != torch.bfloat16:
            self._vla = self._vla.to(self._dtype)
            self._proprio_projector = self._proprio_projector.to(self._dtype)
            self._action_head = self._action_head.to(self._dtype)

    @property
    def norm_keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._vla.norm_stats))

    def infer(self, observation: dict) -> dict:
        import torch
        from experiments.robot import openvla_utils
        from safety_guard.openvla_oft import process_libero_action_chunk, resolve_unnorm_key

        if observation.get("policy/reset"):
            seed = int(observation.get("policy/episode_seed", 0))
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        benchmark = str(observation["policy/benchmark"])
        unnorm_key = resolve_unnorm_key(
            benchmark,
            self._vla.norm_stats,
            libero_90_unnorm_key=self._libero_90_unnorm_key,
        )
        cfg = replace(self._cfg, unnorm_key=unnorm_key)
        vla_observation = {
            "full_image": np.asarray(observation["observation/image"]),
            "wrist_image": np.asarray(observation["observation/wrist_image"]),
            "state": np.asarray(observation["observation/state"], dtype=np.float32).copy(),
        }
        images = [vla_observation["full_image"], vla_observation["wrist_image"]]
        images = openvla_utils.prepare_images_for_vla(images, cfg)
        prompt = f'In: What action should the robot take to {str(observation["prompt"]).lower()}?\nOut:'
        inputs = self._processor(prompt, images[0]).to(openvla_utils.DEVICE, dtype=self._dtype)
        wrist_inputs = self._processor(prompt, images[1]).to(openvla_utils.DEVICE, dtype=self._dtype)
        inputs["pixel_values"] = torch.cat([inputs["pixel_values"], wrist_inputs["pixel_values"]], dim=1)
        proprio_stats = self._vla.norm_stats[unnorm_key]["proprio"]
        proprio = openvla_utils.normalize_proprio(vla_observation["state"], proprio_stats)
        with torch.inference_mode():
            actions, _ = self._vla.predict_action(
                **inputs,
                unnorm_key=unnorm_key,
                do_sample=False,
                proprio=proprio,
                proprio_projector=self._proprio_projector,
                action_head=self._action_head,
                use_film=False,
            )
        return {"actions": process_libero_action_chunk(np.asarray(actions))}


def _pack_array(value):
    if isinstance(value, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": value.tobytes(),
            b"dtype": value.dtype.str,
            b"shape": value.shape,
        }
    if isinstance(value, np.generic):
        return {b"__npgeneric__": True, b"data": value.item(), b"dtype": value.dtype.str}
    raise TypeError(f"Cannot msgpack value of type {type(value)!r}")


def _unpack_array(value):
    if b"__ndarray__" in value:
        return np.ndarray(buffer=value[b"data"], dtype=np.dtype(value[b"dtype"]), shape=value[b"shape"])
    if b"__npgeneric__" in value:
        return np.dtype(value[b"dtype"]).type(value[b"data"])
    return value


packb = functools.partial(msgpack.packb, default=_pack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_array)


class PolicyServer:
    def __init__(self, policy: OpenVLAOFTPolicy, host: str, port: int, metadata: dict) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata

    async def _handler(self, websocket) -> None:
        await websocket.send(packb(self._metadata))
        previous_total = None
        while True:
            try:
                start = time.monotonic()
                observation = unpackb(await websocket.recv())
                infer_start = time.monotonic()
                response = self._policy.infer(observation)
                response["server_timing"] = {"infer_ms": (time.monotonic() - infer_start) * 1000.0}
                if previous_total is not None:
                    response["server_timing"]["prev_total_ms"] = previous_total * 1000.0
                await websocket.send(packb(response))
                previous_total = time.monotonic() - start
            except websockets.ConnectionClosed:
                return
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(code=1011, reason="OpenVLA-OFT inference failed")
                raise

    async def _run(self) -> None:
        async with websockets.asyncio.server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
        ) as server:
            logging.info("OpenVLA-OFT websocket server listening on %s:%d", self._host, self._port)
            await server.serve_forever()

    def serve_forever(self) -> None:
        asyncio.run(self._run())


def main() -> None:
    args = parse_args()
    if args.load_in_8bit and args.load_in_4bit:
        raise ValueError("Choose at most one quantization mode")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    disable_tensorflow_gpu()
    configure_imports(args.openvla_root)
    os.chdir(args.openvla_root)

    cfg = OpenVLAConfig(
        pretrained_checkpoint=str(args.checkpoint),
        unnorm_key="libero_10_no_noops",
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
    )
    policy = OpenVLAOFTPolicy(cfg, args.libero_90_unnorm_key, args.inference_dtype)
    metadata = {
        "policy_backend": "openvla-oft",
        "checkpoint": str(args.checkpoint),
        "action_chunk_size": 8,
        "norm_keys": policy.norm_keys,
        "libero_90_unnorm_key": args.libero_90_unnorm_key,
        "inference_dtype": args.inference_dtype,
    }
    PolicyServer(policy, host=args.host, port=args.port, metadata=metadata).serve_forever()


if __name__ == "__main__":
    main()
