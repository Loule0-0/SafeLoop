from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .risk import RiskVector


def head_values_to_risk(
    p0_logit: float,
    t0_norm: float,
    p1_logit: float,
    t1_norm: float,
) -> RiskVector:
    return RiskVector(
        body_probability=_sigmoid(float(p0_logit)),
        body_tth=float(np.clip(t0_norm, 0.0, 1.0)),
        object_probability=_sigmoid(float(p1_logit)),
        object_tth=float(np.clip(t1_norm, 0.0, 1.0)),
    )


@dataclass
class QwenHeadRiskPredictor:
    model_dir: Path
    head_path: Path
    device: str = "cuda"
    torch_dtype: str = "float16"

    def __post_init__(self) -> None:
        self.model_dir = Path(self.model_dir)
        self.head_path = Path(self.head_path)
        self._model = None
        self._processor = None
        self._head = None

    def predict(self, observation, proposed_action=None, instruction: str | None = None) -> RiskVector:
        if not isinstance(observation, dict):
            raise ValueError("QwenHeadRiskPredictor expects an observation dict")
        text = observation.get("qwen_text") or observation.get("text")
        image_paths = observation.get("qwen_images") or observation.get("image_paths")
        if text is None or image_paths is None:
            raise ValueError("observation must contain qwen_text/text and qwen_images/image_paths")
        return self.predict_prompt(str(text), list(image_paths))

    def predict_prompt(self, text: str, image_paths: Sequence[str | Path]) -> RiskVector:
        self._ensure_loaded()
        import torch
        from PIL import Image

        images = [Image.open(path).convert("RGB") for path in image_paths]
        inputs = self._processor(text=[text], images=[images], padding=True, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            hidden = _extract_last_token_features(self._model, inputs)
            outputs = self._head(hidden.float())

        return head_values_to_risk(
            p0_logit=float(outputs["p0"].squeeze().detach().float().cpu()),
            t0_norm=float(outputs["t0"].squeeze().detach().float().cpu()),
            p1_logit=float(outputs["p1"].squeeze().detach().float().cpu()),
            t1_norm=float(outputs["t1"].squeeze().detach().float().cpu()),
        )

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoProcessor

        dtype = getattr(torch, self.torch_dtype)
        model = AutoModel.from_pretrained(self.model_dir, torch_dtype=dtype, trust_remote_code=True).to(self.device)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        processor = AutoProcessor.from_pretrained(self.model_dir, trust_remote_code=True)
        head = SafetyHead(model.config.hidden_size).to(self.device)
        state = torch.load(self.head_path, map_location=self.device)
        head.load_state_dict(state)
        head.eval()

        self._model = model
        self._processor = processor
        self._head = head


class SafetyHead:
    def __init__(self, hidden_size: int):
        import torch.nn as nn

        class _Head(nn.Module):
            def __init__(self, size: int):
                super().__init__()
                self.shared_layer = nn.Sequential(nn.Linear(size, size), nn.ReLU(), nn.Dropout(0.1))
                self.p_collision = nn.Linear(size, 1)
                self.p_item_damage = nn.Linear(size, 1)
                self.T_collision = nn.Linear(size, 1)
                self.T_item_damage = nn.Linear(size, 1)

            def forward(self, hidden_states):
                import torch.nn.functional as functional

                shared = self.shared_layer(hidden_states)
                return {
                    "p0": self.p_collision(shared),
                    "t0": functional.softplus(self.T_collision(shared)),
                    "p1": self.p_item_damage(shared),
                    "t1": functional.softplus(self.T_item_damage(shared)),
                }

        self._module = _Head(hidden_size)

    def __getattr__(self, name):
        return getattr(self._module, name)


def _extract_last_token_features(model, inputs):
    import torch

    outputs = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        output_hidden_states=True,
        return_dict=True,
    )
    last_hidden_states = outputs.hidden_states[-1]
    last_token_indices = inputs["attention_mask"].sum(dim=1) - 1
    batch_indices = torch.arange(last_hidden_states.shape[0], device=last_hidden_states.device)
    return last_hidden_states[batch_indices, last_token_indices]


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)
