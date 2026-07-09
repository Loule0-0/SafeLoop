from __future__ import annotations

import json
import os
import math
import random
import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .risk import RiskVector

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError:  # pragma: no cover - lets data utilities run without torch.
    torch = None
    nn = None
    F = None

QWEN_IMAGE_TOKEN = "<|vision_start|><|image_pad|><|vision_end|>"


def qwenize_image_placeholders(text: str) -> str:
    """Convert dataset-level image placeholders into Qwen2.5-VL visual tokens."""
    return str(text).replace("<image>", QWEN_IMAGE_TOKEN)


def parse_tau_and_step(text: str) -> tuple[int, int]:
    tau_match = re.search(r"\btau=(\d+)", text)
    step_match = re.search(r"\bcurrent_global_step=(\d+)", text)
    tau = int(tau_match.group(1)) if tau_match else 50
    step = int(step_match.group(1)) if step_match else 0
    return tau, step


def parse_future_labels(sample: dict) -> dict[str, float]:
    user_text = sample["messages"][0]["content"]
    explicit_labels = sample.get("labels") or {}
    assistant_text = sample["messages"][1]["content"]
    tau, _ = parse_tau_and_step(user_text)
    if {"future_body", "future_object"}.issubset(explicit_labels):
        future_body = float(explicit_labels.get("future_body", 0.0))
        future_object = float(explicit_labels.get("future_object", 0.0))
        future_body_tth = float(explicit_labels.get("future_body_tth", 1.0))
        future_object_tth = float(explicit_labels.get("future_object_tth", 1.0))
        return {
            "tau": float(tau),
            "future_body": future_body,
            "future_object": future_object,
            "future_body_tth": future_body_tth,
            "future_object_tth": future_object_tth,
        }
    lines = [line.strip() for line in assistant_text.splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError("assistant label must contain two non-empty lines")
    p_body, t_body = _parse_label_pair(lines[0])
    p_obj, t_obj = _parse_label_pair(lines[1])
    return {
        "tau": float(tau),
        "future_body": float(p_body),
        "future_object": float(p_obj),
        "future_body_tth": _normalize_future_tth(p_body, t_body, tau),
        "future_object_tth": _normalize_future_tth(p_obj, t_obj, tau),
    }


def current_labels_from_marked_steps(
    current_step: int,
    marked_steps: Sequence[Sequence[int]],
    tolerance: int = 5,
) -> tuple[float, float]:
    body = 0.0
    obj = 0.0
    for marked_step, hazard_type in marked_steps:
        if abs(int(marked_step) - int(current_step)) <= tolerance:
            if int(hazard_type) == 0:
                body = 1.0
            elif int(hazard_type) == 1:
                obj = 1.0
    return body, obj


def load_marked_steps(data_root: Path | str) -> dict[str, list[tuple[int, int]]]:
    root = Path(data_root)
    marked_by_run: dict[str, list[tuple[int, int]]] = {}
    for metadata_path in root.glob("full_trajectory_pi0/*/trajectory_metadata.json"):
        run_id = "/".join(metadata_path.relative_to(root).parts[:2])
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
        marked_by_run[run_id] = [
            (int(step), int(hazard_type))
            for step, hazard_type in data.get("marked_steps") or []
        ]
    return marked_by_run


def build_group_split(
    samples: Sequence[dict],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1")
    runs = sorted({_sample_run_id(sample) for sample in samples})
    if len(runs) < 2:
        raise ValueError("at least two run groups are required")
    rng = random.Random(seed)
    rng.shuffle(runs)
    val_run_count = max(1, min(len(runs) - 1, int(round(len(runs) * val_ratio))))
    val_runs = set(runs[:val_run_count])
    train_indices = [
        index for index, sample in enumerate(samples) if _sample_run_id(sample) not in val_runs
    ]
    val_indices = [
        index for index, sample in enumerate(samples) if _sample_run_id(sample) in val_runs
    ]
    return train_indices, val_indices


@dataclass(frozen=True)
class SafetySample:
    text: str
    image_paths: tuple[Path, ...]
    run_id: str
    current_step: int
    future_body: float
    future_body_tth: float
    future_object: float
    future_object_tth: float
    current_body: float
    current_object: float


class SafetyMultitaskDataset:
    def __init__(
        self,
        jsonl_path: Path | str,
        data_root: Path | str,
        current_tolerance: int = 5,
        max_samples: int | None = None,
    ) -> None:
        self.jsonl_path = Path(jsonl_path)
        self.data_root = Path(data_root)
        self.current_tolerance = int(current_tolerance)
        self.marked_steps = load_marked_steps(self.data_root)
        self.samples = self._load_samples(max_samples=max_samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> SafetySample:
        return self.samples[index]

    def _load_samples(self, max_samples: int | None) -> list[SafetySample]:
        samples: list[SafetySample] = []
        with self.jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                raw = json.loads(line)
                future = parse_future_labels(raw)
                text = qwenize_image_placeholders(raw["messages"][0]["content"])
                _, current_step = parse_tau_and_step(raw["messages"][0]["content"])
                image_paths = tuple(self.data_root / str(path).lstrip("/") for path in raw["images"])
                if os.environ.get("SAFELOOP_SKIP_IMAGE_CHECK", "0") != "1":
                    missing = [path for path in image_paths if not path.exists()]
                    if missing:
                        raise FileNotFoundError(f"missing images for sample: {missing[:3]}")
                run_id = "/".join(str(raw["images"][0]).lstrip("/").split("/")[:2])
                current_body, current_object = current_labels_from_marked_steps(
                    current_step=current_step,
                    marked_steps=self.marked_steps.get(run_id, []),
                    tolerance=self.current_tolerance,
                )
                explicit_labels = raw.get("labels") or {}
                if "current_body" in explicit_labels:
                    current_body = float(explicit_labels["current_body"])
                if "current_object" in explicit_labels:
                    current_object = float(explicit_labels["current_object"])
                samples.append(
                    SafetySample(
                        text=text,
                        image_paths=image_paths,
                        run_id=run_id,
                        current_step=current_step,
                        future_body=future["future_body"],
                        future_body_tth=future["future_body_tth"],
                        future_object=future["future_object"],
                        future_object_tth=future["future_object_tth"],
                        current_body=current_body,
                        current_object=current_object,
                    )
                )
                if max_samples is not None and len(samples) >= max_samples:
                    break
        return samples


@dataclass(frozen=True)
class MultitaskSafetyOutput:
    risk: RiskVector
    current_body_probability: float
    current_object_probability: float

    def is_current_hazard(self, threshold: float = 0.5) -> bool:
        return (
            self.current_body_probability >= threshold
            or self.current_object_probability >= threshold
        )


def multitask_values_to_output(
    future_body_logit: float,
    future_body_tth: float,
    future_object_logit: float,
    future_object_tth: float,
    current_body_logit: float,
    current_object_logit: float,
) -> MultitaskSafetyOutput:
    return MultitaskSafetyOutput(
        risk=RiskVector(
            body_probability=_sigmoid(float(future_body_logit)),
            body_tth=_clip01(float(future_body_tth)),
            object_probability=_sigmoid(float(future_object_logit)),
            object_tth=_clip01(float(future_object_tth)),
        ),
        current_body_probability=_sigmoid(float(current_body_logit)),
        current_object_probability=_sigmoid(float(current_object_logit)),
    )


class QwenMultitaskSafetyPredictor:
    def __init__(
        self,
        model_dir: Path | str,
        checkpoint_path: Path | str,
        lora_adapter_path: Path | str | None = None,
        device: str = "cuda",
        torch_dtype: str = "bfloat16",
        history_length: int = 3,
        tau: int = 50,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.checkpoint_path = Path(checkpoint_path)
        self.lora_adapter_path = Path(lora_adapter_path) if lora_adapter_path is not None else None
        self.device = device
        self.torch_dtype = torch_dtype
        self.history_length = int(history_length)
        self.tau = int(tau)
        self._step_index = 0
        self._live_history = deque(maxlen=self.history_length)
        self._model = None
        self._processor = None
        self._head = None

    def predict_prompt(self, text: str, image_paths: Sequence[str | Path]) -> MultitaskSafetyOutput:
        from PIL import Image

        images = [Image.open(path).convert("RGB") for path in image_paths]
        return self.predict_prompt_images(text, images)

    def predict_prompt_images(self, text: str, images: Sequence[object]) -> MultitaskSafetyOutput:
        self._ensure_loaded()
        inputs = self._processor(
            text=[qwenize_image_placeholders(text)],
            images=[images],
            padding=True,
            return_tensors="pt",
        )
        inputs = move_tensors_to_device(inputs, self.device)
        with torch.no_grad():
            hidden = extract_last_token_features(self._model, inputs)
            predictions = self._head(hidden)
        return _single_prediction_to_output(predictions)

    def predict(self, observation, proposed_action=None, instruction: str | None = None) -> MultitaskSafetyOutput:
        frame = live_observation_to_frame(observation)
        self._live_history.append(frame)
        text, images = build_live_qwen_prompt(
            history=list(self._live_history),
            instruction=instruction,
            proposed_action=proposed_action,
            current_global_step=self._step_index,
            history_length=self.history_length,
            tau=self.tau,
        )
        self._step_index += 1
        return self.predict_prompt_images(text, images)

    def reset(self) -> None:
        self._step_index = 0
        self._live_history.clear()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        if torch is None:
            raise ModuleNotFoundError("torch is required for QwenMultitaskSafetyPredictor")
        from transformers import AutoProcessor

        try:
            from transformers import AutoModelForImageTextToText

            model_cls = AutoModelForImageTextToText
        except ImportError:
            from transformers import Qwen2_5_VLForConditionalGeneration

            model_cls = Qwen2_5_VLForConditionalGeneration

        dtype = getattr(torch, self.torch_dtype)
        model = model_cls.from_pretrained(
            self.model_dir,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        lora_adapter_path = self.lora_adapter_path or _sibling_lora_adapter_path(self.checkpoint_path)
        if lora_adapter_path is not None:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, lora_adapter_path)
        model = model.to(self.device)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        hidden_size = int(checkpoint.get("hidden_size") or _config_hidden_size(model.config))
        neck_size = int(checkpoint.get("neck_size", 512))
        head = MultitaskSafetyHead(hidden_size=hidden_size, neck_size=neck_size).to(self.device)
        head.load_state_dict(checkpoint["head_state_dict"])
        head.eval()

        self._model = model
        self._processor = AutoProcessor.from_pretrained(self.model_dir, trust_remote_code=True)
        self._head = head


class MultitaskSafetyHead(nn.Module if nn is not None else object):
    """Shared neck with future risk and current-hazard classifier heads."""

    def __init__(self, hidden_size: int, neck_size: int = 512, dropout: float = 0.1) -> None:
        if nn is None:
            raise ModuleNotFoundError("torch is required to create MultitaskSafetyHead")
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.neck_size = int(neck_size)
        self.neck = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.neck_size),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.risk_logits = nn.Linear(self.neck_size, 2)
        self.risk_tth = nn.Linear(self.neck_size, 2)
        self.current_logits = nn.Linear(self.neck_size, 2)
        self._init_weights()

    def forward(self, hidden_states):
        features = self.neck(hidden_states.float())
        return {
            "risk_logits": self.risk_logits(features),
            "risk_tth": torch.sigmoid(self.risk_tth(features)),
            "current_logits": self.current_logits(features),
        }

    def _init_weights(self) -> None:
        for module in [self.risk_logits, self.risk_tth, self.current_logits]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)


class SafetyMultitaskLoss(nn.Module if nn is not None else object):
    def __init__(
        self,
        future_pos_weight=None,
        current_pos_weight=None,
        future_bce_weight: float = 1.0,
        future_tth_weight: float = 1.0,
        current_bce_weight: float = 1.0,
        body_weight: float = 1.0,
        object_weight: float = 1.0,
    ) -> None:
        if nn is None:
            raise ModuleNotFoundError("torch is required to create SafetyMultitaskLoss")
        super().__init__()
        self.future_bce_weight = float(future_bce_weight)
        self.future_tth_weight = float(future_tth_weight)
        self.current_bce_weight = float(current_bce_weight)
        self.register_buffer(
            "class_weight",
            torch.as_tensor([float(body_weight), float(object_weight)], dtype=torch.float32),
        )
        if future_pos_weight is not None:
            self.register_buffer("future_pos_weight", torch.as_tensor(future_pos_weight, dtype=torch.float32))
        else:
            self.future_pos_weight = None
        if current_pos_weight is not None:
            self.register_buffer("current_pos_weight", torch.as_tensor(current_pos_weight, dtype=torch.float32))
        else:
            self.current_pos_weight = None

    def forward(self, predictions: dict, labels: dict):
        future_labels = labels["future_labels"].float()
        future_tth = labels["future_tth"].float()
        current_labels = labels["current_labels"].float()
        future_pos_weight = _buffer_to_device(self.future_pos_weight, future_labels.device)
        current_pos_weight = _buffer_to_device(self.current_pos_weight, current_labels.device)
        class_weight = _buffer_to_device(self.class_weight, future_labels.device)

        future_bce = _weighted_mean(
            F.binary_cross_entropy_with_logits(
                predictions["risk_logits"].float(),
                future_labels,
                pos_weight=future_pos_weight,
                reduction="none",
            ),
            class_weight,
        )
        future_tth_loss = _weighted_mean(
            F.smooth_l1_loss(
                predictions["risk_tth"].float(),
                future_tth,
                reduction="none",
            ),
            class_weight,
        )
        current_bce = _weighted_mean(
            F.binary_cross_entropy_with_logits(
                predictions["current_logits"].float(),
                current_labels,
                pos_weight=current_pos_weight,
                reduction="none",
            ),
            class_weight,
        )
        total = (
            self.future_bce_weight * future_bce
            + self.future_tth_weight * future_tth_loss
            + self.current_bce_weight * current_bce
        )
        return total, {
            "future_bce": float(future_bce.detach().cpu()),
            "future_tth": float(future_tth_loss.detach().cpu()),
            "current_bce": float(current_bce.detach().cpu()),
            "total": float(total.detach().cpu()),
        }


def _weighted_mean(loss, class_weight):
    if class_weight is None:
        return loss.mean()
    return (loss * class_weight.view(1, -1)).mean()


def compute_pos_weights(samples: Sequence[SafetySample], cap: float = 30.0) -> dict[str, list[float]]:
    if not samples:
        raise ValueError("samples must not be empty")
    future_pos = [0.0, 0.0]
    current_pos = [0.0, 0.0]
    for sample in samples:
        future_pos[0] += float(sample.future_body)
        future_pos[1] += float(sample.future_object)
        current_pos[0] += float(sample.current_body)
        current_pos[1] += float(sample.current_object)
    total = float(len(samples))
    return {
        "future": [_pos_weight(pos, total, cap) for pos in future_pos],
        "current": [_pos_weight(pos, total, cap) for pos in current_pos],
    }


def current_positive_sample_weights(
    samples: Sequence[SafetySample],
    positive_weight: float = 1.0,
    body_positive_weight: float | None = None,
    object_positive_weight: float | None = None,
) -> list[float]:
    body_weight = float(positive_weight if body_positive_weight is None else body_positive_weight)
    object_weight = float(positive_weight if object_positive_weight is None else object_positive_weight)
    if positive_weight < 1.0 or body_weight < 1.0 or object_weight < 1.0:
        raise ValueError("positive sample weights must be >= 1.0")
    weights: list[float] = []
    for sample in samples:
        weight = 1.0
        if float(sample.current_body) > 0.0:
            weight = max(weight, body_weight)
        if float(sample.current_object) > 0.0:
            weight = max(weight, object_weight)
        weights.append(float(weight))
    return weights


def compute_multitask_metrics(predictions: dict, labels: dict, threshold: float = 0.5) -> dict[str, float]:
    if torch is None:
        raise ModuleNotFoundError("torch is required to compute metrics")
    with torch.no_grad():
        future_prob = torch.sigmoid(predictions["risk_logits"].float())
        current_prob = torch.sigmoid(predictions["current_logits"].float())
        future_pred = (future_prob >= threshold).float()
        current_pred = (current_prob >= threshold).float()
        future_labels = labels["future_labels"].float()
        current_labels = labels["current_labels"].float()
        risk_tth = predictions["risk_tth"].float()
        future_tth = labels["future_tth"].float()

        metrics = {
            "future_body_acc": _binary_accuracy(future_pred[:, 0], future_labels[:, 0]),
            "future_object_acc": _binary_accuracy(future_pred[:, 1], future_labels[:, 1]),
            "current_body_acc": _binary_accuracy(current_pred[:, 0], current_labels[:, 0]),
            "current_object_acc": _binary_accuracy(current_pred[:, 1], current_labels[:, 1]),
            "current_body_recall": _binary_recall(current_pred[:, 0], current_labels[:, 0]),
            "current_object_recall": _binary_recall(current_pred[:, 1], current_labels[:, 1]),
            "current_body_precision": _binary_precision(current_pred[:, 0], current_labels[:, 0]),
            "current_object_precision": _binary_precision(current_pred[:, 1], current_labels[:, 1]),
            "current_body_ap": _average_precision(current_prob[:, 0], current_labels[:, 0]),
            "current_object_ap": _average_precision(current_prob[:, 1], current_labels[:, 1]),
            "current_body_prob_pos_mean": _prob_mean(current_prob[:, 0], current_labels[:, 0], positive=True),
            "current_body_prob_neg_mean": _prob_mean(current_prob[:, 0], current_labels[:, 0], positive=False),
            "current_object_prob_pos_mean": _prob_mean(current_prob[:, 1], current_labels[:, 1], positive=True),
            "current_object_prob_neg_mean": _prob_mean(current_prob[:, 1], current_labels[:, 1], positive=False),
            "future_tth_mae": float(torch.mean(torch.abs(risk_tth - future_tth)).detach().cpu()),
        }
        return metrics


def compute_multitask_metrics_at_thresholds(
    predictions: dict,
    labels: dict,
    thresholds: Sequence[float] = (0.5, 0.3, 0.2),
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for index, threshold in enumerate(thresholds):
        threshold_metrics = compute_multitask_metrics(predictions, labels, threshold=threshold)
        if index == 0:
            metrics.update(threshold_metrics)
            continue
        suffix = _threshold_suffix(threshold)
        for key, value in threshold_metrics.items():
            if key.endswith("_ap") or "prob_" in key or key == "future_tth_mae":
                continue
            metrics[f"{key}_{suffix}"] = value
    return metrics


def collate_multitask_batch(batch: Sequence[SafetySample], processor):
    if torch is None:
        raise ModuleNotFoundError("torch is required to collate training batches")
    from PIL import Image

    texts = [sample.text for sample in batch]
    images = [
        [Image.open(path).convert("RGB") for path in sample.image_paths]
        for sample in batch
    ]
    inputs = processor(text=texts, images=images, padding=True, return_tensors="pt")
    labels = {
        "future_labels": torch.tensor(
            [[sample.future_body, sample.future_object] for sample in batch],
            dtype=torch.float32,
        ),
        "future_tth": torch.tensor(
            [[sample.future_body_tth, sample.future_object_tth] for sample in batch],
            dtype=torch.float32,
        ),
        "current_labels": torch.tensor(
            [[sample.current_body, sample.current_object] for sample in batch],
            dtype=torch.float32,
        ),
    }
    return inputs, labels


@dataclass(frozen=True)
class LiveQwenFrame:
    images: tuple[object, object]
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    joint_torques: np.ndarray


def live_observation_to_frame(observation) -> LiveQwenFrame:
    from PIL import Image
    import numpy as np

    if not isinstance(observation, dict):
        observation = {}
    agent = _pil_from_observation_image(
        _first_present(observation, ("agentview_image", "observation/image", "image")),
    )
    wrist = _pil_from_observation_image(
        _first_present(observation, ("robot0_eye_in_hand_image", "observation/wrist_image", "wrist_image")),
    )
    if agent is None:
        agent = Image.new("RGB", (224, 224), color=(0, 0, 0))
    if wrist is None:
        wrist = Image.new("RGB", agent.size, color=(0, 0, 0))

    joint_pos = _observation_array(observation, ("robot0_joint_pos", "joint_pos"), 7)
    joint_vel = _observation_array(observation, ("robot0_joint_vel", "joint_vel"), 7)
    joint_torques = _observation_array(observation, ("robot0_joint_torques", "joint_torques"), 9)
    return LiveQwenFrame(
        images=(agent, wrist),
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        joint_torques=joint_torques,
    )


def build_live_qwen_prompt(
    history: Sequence[LiveQwenFrame],
    instruction: str | None = None,
    proposed_action=None,
    current_global_step: int = 0,
    history_length: int = 3,
    tau: int = 50,
) -> tuple[str, list[object]]:
    if not history:
        history = [live_observation_to_frame({})]
    frames = list(history)[-history_length:]
    while len(frames) < history_length:
        frames.insert(0, frames[0])

    lines = [
        f"H={history_length}, tau={tau}, current_global_step={int(current_global_step)}",
        "Hazard types: 0=collision, 1=item_damage",
        "For each of the past H steps (oldest->newest), we provide:",
    ]
    images: list[object] = []
    for frame in frames:
        images.extend(frame.images)
        lines.extend(
            [
                f"- images: {QWEN_IMAGE_TOKEN} {QWEN_IMAGE_TOKEN}",
                f"- robot0_joint_pos: {_format_float_list(frame.joint_pos)}",
                f"- robot0_joint_vel: {_format_float_list(frame.joint_vel)}",
                f"- robot0_joint_torques: {_format_float_list(frame.joint_torques)}",
            ]
        )
    if instruction:
        lines.append(f"Task instruction: {instruction}")
    if proposed_action is not None:
        lines.append(f"Proposed action: {_format_float_list(np.asarray(proposed_action, dtype=np.float32).reshape(-1))}")
    lines.append("Predict collision and item_damage as '<probability> <time_to_hazard_steps>'.")
    return "\n".join(lines), images


def move_tensors_to_device(batch: dict, device: str):
    if torch is None:
        raise ModuleNotFoundError("torch is required to move tensors")
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in batch.items()
    }


def extract_last_token_features(model, inputs: dict):
    if torch is None:
        raise ModuleNotFoundError("torch is required to extract features")
    outputs = model(
        **inputs,
        output_hidden_states=True,
        return_dict=True,
        use_cache=False,
    )
    last_hidden = outputs.hidden_states[-1]
    last_token_indices = inputs["attention_mask"].sum(dim=1) - 1
    batch_indices = torch.arange(last_hidden.shape[0], device=last_hidden.device)
    return last_hidden[batch_indices, last_token_indices]


def _first_present(mapping: dict, keys: Sequence[str]):
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _pil_from_observation_image(value):
    if value is None:
        return None
    from PIL import Image
    import numpy as np

    if isinstance(value, Image.Image):
        return value.convert("RGB")
    array = np.asarray(value)
    if array.ndim == 4:
        array = array[0]
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    if array.dtype != np.uint8:
        if float(np.nanmax(array)) <= 1.0:
            array = np.clip(array * 255.0, 0, 255)
        array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(array[..., :3]).convert("RGB")


def _observation_array(observation: dict, keys: Sequence[str], target_dim: int) -> np.ndarray:
    value = _first_present(observation, keys)
    if value is None:
        array = np.zeros(0, dtype=np.float32)
    else:
        array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.shape[0] == target_dim:
        return array.astype(np.float32)
    fitted = np.zeros(target_dim, dtype=np.float32)
    fitted[: min(target_dim, array.shape[0])] = array[:target_dim]
    return fitted


def _format_float_list(values: Sequence[float]) -> str:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    return "[" + ", ".join(f"{float(value):.6f}" for value in array) + "]"


def _single_prediction_to_output(predictions: dict) -> MultitaskSafetyOutput:
    risk_logits = predictions["risk_logits"].detach().float().cpu()[0]
    risk_tth = predictions["risk_tth"].detach().float().cpu()[0]
    current_logits = predictions["current_logits"].detach().float().cpu()[0]
    return multitask_values_to_output(
        future_body_logit=float(risk_logits[0]),
        future_body_tth=float(risk_tth[0]),
        future_object_logit=float(risk_logits[1]),
        future_object_tth=float(risk_tth[1]),
        current_body_logit=float(current_logits[0]),
        current_object_logit=float(current_logits[1]),
    )


def _buffer_to_device(value, device):
    if value is None:
        return None
    return value.to(device)


def _pos_weight(pos_count: float, total: float, cap: float) -> float:
    if pos_count <= 0:
        return 1.0
    neg_count = max(total - pos_count, 0.0)
    return min(math.sqrt(neg_count / pos_count), float(cap))


def _binary_accuracy(pred, target) -> float:
    return float((pred == target).float().mean().detach().cpu())


def _binary_recall(pred, target) -> float:
    positives = target.sum()
    if float(positives) <= 0.0:
        return 0.0
    true_pos = ((pred == 1.0) & (target == 1.0)).float().sum()
    return float((true_pos / positives).detach().cpu())


def _binary_precision(pred, target) -> float:
    predicted_pos = pred.sum()
    if float(predicted_pos) <= 0.0:
        return 0.0
    true_pos = ((pred == 1.0) & (target == 1.0)).float().sum()
    return float((true_pos / predicted_pos).detach().cpu())


def _average_precision(prob, target) -> float:
    positives = target.sum()
    if float(positives) <= 0.0:
        return 0.0
    order = torch.argsort(prob, descending=True)
    sorted_target = target[order].float()
    true_positives = torch.cumsum(sorted_target, dim=0)
    ranks = torch.arange(1, sorted_target.numel() + 1, device=sorted_target.device, dtype=torch.float32)
    precision_at_rank = true_positives / ranks
    ap = (precision_at_rank * sorted_target).sum() / positives
    return float(ap.detach().cpu())


def _prob_mean(prob, target, positive: bool) -> float:
    mask = target > 0.5 if positive else target <= 0.5
    if int(mask.sum().detach().cpu()) == 0:
        return 0.0
    return float(prob[mask].mean().detach().cpu())


def _threshold_suffix(threshold: float) -> str:
    return f"t{int(round(float(threshold) * 100)):03d}"


def _sibling_lora_adapter_path(checkpoint_path: Path) -> Path | None:
    adapter_path = Path(checkpoint_path).parent / "lora_adapter"
    if adapter_path.exists():
        return adapter_path
    return None


def _parse_label_pair(line: str) -> tuple[float, float]:
    parts = line.split()
    if len(parts) < 2:
        raise ValueError(f"expected '<p> <tth>', got {line!r}")
    return float(parts[0]), float(parts[1])


def _sample_run_id(sample: dict | SafetySample) -> str:
    if isinstance(sample, dict):
        return str(sample["run_id"])
    return str(sample.run_id)


def _config_hidden_size(config) -> int:
    hidden_size = getattr(config, "hidden_size", None)
    if hidden_size is None and getattr(config, "text_config", None) is not None:
        hidden_size = getattr(config.text_config, "hidden_size", None)
    if hidden_size is None:
        raise ValueError("Could not resolve hidden size from Qwen config")
    return int(hidden_size)


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _clip01(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def _normalize_future_tth(probability: float, tth_steps: float, tau: int) -> float:
    if tau <= 0:
        raise ValueError("tau must be positive")
    if probability <= 0.0 or tth_steps < 0:
        return 1.0
    return min(max(float(tth_steps), 0.0), float(tau)) / float(tau)
