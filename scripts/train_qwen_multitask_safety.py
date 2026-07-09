from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from datetime import datetime
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
    current_positive_sample_weights,
    extract_last_token_features,
    move_tensors_to_device,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Qwen2.5-VL multitask SafeLoop heads.")
    parser.add_argument("--model-dir", "--model-path", dest="model_dir", type=Path, required=True)
    parser.add_argument("--data", "--data-jsonl", dest="data", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", "--output-dir", dest="out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--lr", "--learning-rate", dest="lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--train-mode", choices=["frozen", "lora"], default="frozen")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--future-window", type=int, default=None)
    parser.add_argument("--current-tolerance", type=int, default=5)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--neck-size", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--future-bce-weight", type=float, default=1.0)
    parser.add_argument("--future-tth-weight", type=float, default=1.0)
    parser.add_argument("--current-bce-weight", type=float, default=1.0)
    parser.add_argument("--body-weight", type=float, default=1.0)
    parser.add_argument("--object-weight", type=float, default=1.0)
    parser.add_argument("--current-positive-sample-weight", type=float, default=1.0)
    parser.add_argument("--current-body-positive-sample-weight", type=float, default=None)
    parser.add_argument("--current-object-positive-sample-weight", type=float, default=None)
    parser.add_argument("--pos-weight-cap", type=float, default=30.0)
    parser.add_argument("--init-head", type=Path, default=None)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=0)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.5, 0.7, 0.8, 0.9, 0.95, 0.3, 0.2],
    )
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--init-lora-adapter", type=Path, default=None)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.freeze_backbone:
        args.train_mode = "frozen"
    if args.grad_accum_steps < 1:
        raise ValueError("--grad-accum-steps must be >= 1")
    if args.save_steps < 0:
        raise ValueError("--save-steps must be >= 0")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
    from tqdm import tqdm
    from transformers import AutoProcessor, get_cosine_schedule_with_warmup

    model_cls = _qwen_model_class()
    _set_seed(args.seed, torch=torch, np=np)
    run_dir = args.out / f"multitask_{args.train_mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)
    dtype = _torch_dtype(args, torch)
    model = model_cls.from_pretrained(
        args.model_dir,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(args.device)
    model.config.use_cache = False

    if args.train_mode == "frozen":
        if args.init_lora_adapter is not None:
            model = _load_lora_adapter(model, args.init_lora_adapter, is_trainable=False)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad = False
    else:
        model = _enable_lora(model, args)
        model.train()

    head = MultitaskSafetyHead(
        hidden_size=_model_hidden_size(model.config),
        neck_size=args.neck_size,
        dropout=args.dropout,
    ).to(args.device)
    if args.init_head:
        checkpoint = torch.load(args.init_head, map_location=args.device)
        state = checkpoint.get("head_state_dict", checkpoint)
        head.load_state_dict(state)

    dataset = SafetyMultitaskDataset(
        args.data,
        data_root=args.data_root,
        current_tolerance=args.current_tolerance,
        max_samples=args.max_samples,
    )
    train_indices, val_indices = build_group_split(dataset.samples, args.val_ratio, args.seed)
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    train_samples = [dataset.samples[index] for index in train_indices]
    pos_weights = compute_pos_weights(train_samples, cap=args.pos_weight_cap)
    train_sampler = None
    train_shuffle = True
    sample_weights = None
    body_sample_weight = (
        args.current_positive_sample_weight
        if args.current_body_positive_sample_weight is None
        else args.current_body_positive_sample_weight
    )
    object_sample_weight = (
        args.current_positive_sample_weight
        if args.current_object_positive_sample_weight is None
        else args.current_object_positive_sample_weight
    )
    if max(args.current_positive_sample_weight, body_sample_weight, object_sample_weight) > 1.0:
        sample_weights = current_positive_sample_weights(
            train_samples,
            positive_weight=args.current_positive_sample_weight,
            body_positive_weight=body_sample_weight,
            object_positive_weight=object_sample_weight,
        )
        train_sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_shuffle = False

    criterion = SafetyMultitaskLoss(
        future_pos_weight=pos_weights["future"],
        current_pos_weight=pos_weights["current"],
        future_bce_weight=args.future_bce_weight,
        future_tth_weight=args.future_tth_weight,
        current_bce_weight=args.current_bce_weight,
        body_weight=args.body_weight,
        object_weight=args.object_weight,
    )
    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=lambda batch: collate_multitask_batch(batch, processor),
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda batch: collate_multitask_batch(batch, processor),
    )

    trainable_params = list(head.parameters()) + [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    optimizer_steps_per_epoch = max(1, (len(train_loader) + args.grad_accum_steps - 1) // args.grad_accum_steps)
    total_steps = max(1, optimizer_steps_per_epoch * args.epochs)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_steps * 0.05)),
        num_training_steps=total_steps,
    )
    amp_dtype = _amp_dtype(args, torch)

    _write_json(
        run_dir / "config.json",
        {
            "args": _jsonable_args(args),
            "dataset_size": len(dataset),
            "train_size": len(train_subset),
            "val_size": len(val_subset),
            "pos_weights": pos_weights,
            "train_runs": sorted({dataset.samples[index].run_id for index in train_indices}),
            "val_runs": sorted({dataset.samples[index].run_id for index in val_indices}),
            "sampler": {
                "current_positive_sample_weight": args.current_positive_sample_weight,
                "current_body_positive_sample_weight": body_sample_weight,
                "current_object_positive_sample_weight": object_sample_weight,
                "enabled": train_sampler is not None,
                "min_weight": min(sample_weights) if sample_weights else 1.0,
                "max_weight": max(sample_weights) if sample_weights else 1.0,
            },
        },
    )
    metrics_path = run_dir / "metrics.csv"
    best_score = -1.0
    global_step = 0

    print(f"run_dir={run_dir}", flush=True)
    print(f"dataset={len(dataset)} train={len(train_subset)} val={len(val_subset)}", flush=True)
    print(f"pos_weights={pos_weights}", flush=True)
    print(
        f"optimizer_steps_per_epoch={optimizer_steps_per_epoch} grad_accum_steps={args.grad_accum_steps}",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        train_parts, global_step = _train_epoch(
            model=model,
            head=head,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=args.device,
            amp_dtype=amp_dtype,
            max_grad_norm=args.max_grad_norm,
            frozen_backbone=(args.train_mode == "frozen"),
            grad_accum_steps=args.grad_accum_steps,
            global_step=global_step,
            save_steps=args.save_steps,
            run_dir=run_dir,
            processor=processor,
            args=args,
            epoch=epoch,
            tqdm=tqdm,
            torch=torch,
        )
        val_parts, val_metrics = _evaluate(
            model=model,
            head=head,
            loader=val_loader,
            criterion=criterion,
            device=args.device,
            amp_dtype=amp_dtype,
            thresholds=tuple(args.thresholds),
            tqdm=tqdm,
            torch=torch,
        )
        score = _selection_score(val_metrics)
        row = {"epoch": epoch, "global_step": global_step, **{f"train_{k}": v for k, v in train_parts.items()}, **{f"val_{k}": v for k, v in val_parts.items()}, **val_metrics}
        _append_metrics(metrics_path, row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

        if epoch % args.save_every == 0:
            _save_checkpoint(run_dir / f"epoch_{epoch:03d}", model, processor, head, args, row)
        if score > best_score:
            best_score = score
            _save_checkpoint(run_dir / "best", model, processor, head, args, row)


def _train_epoch(
    model,
    head,
    loader,
    criterion,
    optimizer,
    scheduler,
    device,
    amp_dtype,
    max_grad_norm,
    frozen_backbone,
    grad_accum_steps,
    global_step,
    save_steps,
    run_dir,
    processor,
    args,
    epoch,
    tqdm,
    torch,
) -> tuple[dict[str, float], int]:
    head.train()
    if frozen_backbone:
        model.eval()
    else:
        model.train()
    totals: dict[str, float] = {}
    micro_steps = 0
    optimizer.zero_grad(set_to_none=True)
    trainable_params = [p for group in optimizer.param_groups for p in group["params"]]
    for batch_index, (inputs, labels) in enumerate(tqdm(loader, desc="train", leave=False), start=1):
        inputs = move_tensors_to_device(inputs, device)
        labels = move_tensors_to_device(labels, device)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            if frozen_backbone:
                with torch.no_grad():
                    hidden = extract_last_token_features(model, inputs)
            else:
                hidden = extract_last_token_features(model, inputs)
            predictions = head(hidden)
            loss, parts = criterion(predictions, labels)
            scaled_loss = loss / float(grad_accum_steps)
        scaled_loss.backward()
        for key, value in parts.items():
            totals[key] = totals.get(key, 0.0) + float(value)
        micro_steps += 1
        should_step = batch_index % grad_accum_steps == 0 or batch_index == len(loader)
        if not should_step:
            continue
        torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1
        if save_steps > 0 and global_step % save_steps == 0:
            _save_checkpoint(
                run_dir / f"step_{global_step:06d}",
                model,
                processor,
                head,
                args,
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    **{f"train_{key}": value / max(micro_steps, 1) for key, value in totals.items()},
                },
                save_processor=False,
            )
    return {key: value / max(micro_steps, 1) for key, value in totals.items()}, global_step


def _evaluate(model, head, loader, criterion, device, amp_dtype, thresholds, tqdm, torch) -> tuple[dict[str, float], dict[str, float]]:
    model.eval()
    head.eval()
    totals: dict[str, float] = {}
    all_predictions: dict[str, list] = {"risk_logits": [], "risk_tth": [], "current_logits": []}
    all_labels: dict[str, list] = {"future_labels": [], "future_tth": [], "current_labels": []}
    steps = 0
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="val", leave=False):
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
    return (
        {key: value / max(steps, 1) for key, value in totals.items()},
        compute_multitask_metrics_at_thresholds(merged_predictions, merged_labels, thresholds=thresholds),
    )


def _qwen_model_class():
    try:
        from transformers import AutoModelForImageTextToText

        return AutoModelForImageTextToText
    except ImportError:
        from transformers import Qwen2_5_VLForConditionalGeneration

        return Qwen2_5_VLForConditionalGeneration


def _enable_lora(model, args):
    try:
        from peft import LoraConfig, PeftModel, get_peft_model
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("PEFT is required for --train-mode lora") from exc
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if args.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if args.init_lora_adapter is not None:
        model = PeftModel.from_pretrained(model, args.init_lora_adapter, is_trainable=True)
        model.print_trainable_parameters()
        return model
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def _load_lora_adapter(model, adapter_path: Path, is_trainable: bool):
    try:
        from peft import PeftModel
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("PEFT is required to load --init-lora-adapter") from exc
    return PeftModel.from_pretrained(model, adapter_path, is_trainable=is_trainable)


def _model_hidden_size(config) -> int:
    hidden_size = getattr(config, "hidden_size", None)
    if hidden_size is None and getattr(config, "text_config", None) is not None:
        hidden_size = getattr(config.text_config, "hidden_size", None)
    if hidden_size is None:
        raise ValueError("Could not resolve Qwen hidden size from model config")
    return int(hidden_size)


def _save_checkpoint(path: Path, model, processor, head, args, metrics: dict, save_processor: bool = True) -> None:
    import torch

    path.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "head_state_dict": head.state_dict(),
            "hidden_size": int(head.hidden_size),
            "neck_size": int(head.neck_size),
            "args": _jsonable_args(args),
            "metrics": metrics,
        },
        path / "multitask_head.pt",
    )
    if save_processor:
        processor.save_pretrained(path / "processor")
    if args.train_mode == "lora" and hasattr(model, "save_pretrained"):
        model.save_pretrained(path / "lora_adapter")


def _append_metrics(path: Path, row: dict) -> None:
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _selection_score(metrics: dict[str, float]) -> float:
    object_f1 = _f1_score(
        precision=metrics.get("current_object_precision", 0.0),
        recall=metrics.get("current_object_recall", 0.0),
    )
    body_f1 = _f1_score(
        precision=metrics.get("current_body_precision", 0.0),
        recall=metrics.get("current_body_recall", 0.0),
    )
    return (
        0.30 * metrics.get("future_body_acc", 0.0)
        + 0.50 * metrics.get("future_object_acc", 0.0)
        + 1.00 * metrics.get("current_object_ap", 0.0)
        + 1.00 * object_f1
        + 0.50 * metrics.get("current_object_precision_t090", 0.0)
        + 0.20 * body_f1
    )


def _f1_score(*, precision: float | None, recall: float | None) -> float:
    precision = float(precision or 0.0)
    recall = float(recall or 0.0)
    denom = precision + recall
    if denom <= 0.0:
        return 0.0
    return float(2.0 * precision * recall / denom)


def _set_seed(seed: int, torch, np) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def _jsonable_args(args: argparse.Namespace) -> dict:
    values = vars(args).copy()
    return {key: str(value) if isinstance(value, Path) else value for key, value in values.items()}


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
