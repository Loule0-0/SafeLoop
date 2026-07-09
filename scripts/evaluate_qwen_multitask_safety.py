from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from safety_guard.qwen_multitask import (
    MultitaskSafetyHead,
    SafetyMultitaskDataset,
    SafetyMultitaskLoss,
    build_group_split,
    collate_multitask_batch,
    compute_multitask_metrics_at_thresholds,
    compute_pos_weights,
    extract_last_token_features,
    move_tensors_to_device,
)
from scripts.train_qwen_multitask_safety import _model_hidden_size, _qwen_model_class


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a Qwen2.5-VL multitask SafeLoop checkpoint.")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--current-tolerance", type=int, default=5)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.5, 0.3, 0.2, 0.1])
    parser.add_argument("--lora-adapter", type=Path, default=None)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    import torch
    from torch.utils.data import DataLoader, Subset
    from tqdm import tqdm
    from transformers import AutoProcessor

    model_cls = _qwen_model_class()
    processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)
    model = model_cls.from_pretrained(
        args.model_dir,
        torch_dtype=_torch_dtype(args, torch),
        trust_remote_code=True,
    ).to(args.device)
    lora_adapter = args.lora_adapter or _sibling_lora_adapter_path(args.checkpoint)
    if lora_adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, lora_adapter).to(args.device)
    model.config.use_cache = False
    model.eval()

    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    head = MultitaskSafetyHead(
        hidden_size=int(checkpoint.get("hidden_size") or _model_hidden_size(model.config)),
        neck_size=int(checkpoint.get("neck_size", 512)),
    ).to(args.device)
    head.load_state_dict(checkpoint["head_state_dict"])
    head.eval()

    dataset = SafetyMultitaskDataset(
        args.data,
        data_root=args.data_root,
        current_tolerance=args.current_tolerance,
        max_samples=args.max_samples,
    )
    train_indices, val_indices = build_group_split(dataset.samples, args.val_ratio, args.seed)
    val_subset = Subset(dataset, val_indices)
    train_samples = [dataset.samples[index] for index in train_indices]
    pos_weights = compute_pos_weights(train_samples)
    criterion = SafetyMultitaskLoss(
        future_pos_weight=pos_weights["future"],
        current_pos_weight=pos_weights["current"],
    )
    loader = DataLoader(
        val_subset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_multitask_batch(batch, processor),
    )
    metrics = _evaluate(
        model=model,
        head=head,
        loader=loader,
        criterion=criterion,
        device=args.device,
        amp_dtype=_amp_dtype(args, torch),
        thresholds=tuple(args.thresholds),
        tqdm=tqdm,
        torch=torch,
    )
    payload = {
        "checkpoint": str(args.checkpoint),
        "lora_adapter": str(lora_adapter) if lora_adapter is not None else None,
        "dataset_size": len(dataset),
        "val_size": len(val_subset),
        "thresholds": args.thresholds,
        "metrics": metrics,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text, flush=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")


def _evaluate(model, head, loader, criterion, device, amp_dtype, thresholds, tqdm, torch) -> dict[str, float]:
    totals: dict[str, float] = {}
    all_predictions: dict[str, list] = {"risk_logits": [], "risk_tth": [], "current_logits": []}
    all_labels: dict[str, list] = {"future_labels": [], "future_tth": [], "current_labels": []}
    steps = 0
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="eval", leave=False):
            inputs = move_tensors_to_device(inputs, device)
            labels = move_tensors_to_device(labels, device)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                hidden = extract_last_token_features(model, inputs)
                predictions = head(hidden)
                _, parts = criterion(predictions, labels)
            for key, value in parts.items():
                totals[key] = totals.get(key, 0.0) + float(value)
            for key in all_predictions:
                all_predictions[key].append(predictions[key].detach().cpu())
            for key in all_labels:
                all_labels[key].append(labels[key].detach().cpu())
            steps += 1
    merged_predictions = {key: torch.cat(value, dim=0) for key, value in all_predictions.items()}
    merged_labels = {key: torch.cat(value, dim=0) for key, value in all_labels.items()}
    metrics = {f"val_{key}": value / max(steps, 1) for key, value in totals.items()}
    metrics.update(compute_multitask_metrics_at_thresholds(merged_predictions, merged_labels, thresholds=thresholds))
    return metrics


def _torch_dtype(args, torch):
    if args.bf16:
        return torch.bfloat16
    if args.fp16:
        return torch.float16
    return torch.float32


def _amp_dtype(args, torch):
    if not str(args.device).startswith("cuda"):
        return None
    if args.bf16:
        return torch.bfloat16
    if args.fp16:
        return torch.float16
    return None


def _sibling_lora_adapter_path(checkpoint: Path) -> Path | None:
    adapter_path = checkpoint.parent / "lora_adapter"
    if adapter_path.exists():
        return adapter_path
    return None


if __name__ == "__main__":
    main()
