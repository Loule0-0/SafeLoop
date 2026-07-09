# SafeLoop Data And Training Pipeline

This guide describes the public, reproducible pipeline for SafeLoop. It intentionally avoids raw experiment logs and machine-specific artifact paths.

## Method Summary

SafeLoop wraps a frozen base manipulation policy with three modules:

1. A short-horizon hazard predictor.
2. A memory of safe rollback anchors.
3. A decision head over `noop`, `record`, and `rollback`.

At runtime, the predictor estimates whether body/stuck or object hazards are likely within a future window. The decision head records safe anchors and triggers rollback when the risk and policy state indicate that intervention is useful. After rollback, the frozen base policy resumes from the safer state.

## Data Collection

Use closed-loop rollout collection to generate predictor samples and decision traces:

```bash
python scripts/evaluate_pi0_safeguard_closed_loop.py \
  --libero-root "$LIBERO_ROOT" \
  --benchmark libero_10 \
  --task-id 0 \
  --episodes 1 \
  --policy-mode websocket \
  --predictor qwen-multitask \
  --qwen-model-path "$QWEN_MODEL" \
  --qwen-head-path /path/to/multitask_head.pt \
  --qwen-rollout-jsonl-out "$SAFELOOP_OUTPUT/qwen_rollout.jsonl" \
  --qwen-rollout-image-root "$SAFELOOP_OUTPUT/qwen_images" \
  --qwen-rollout-stride 30 \
  --qwen-rollout-prehazard-stride 8 \
  --decision-debug-jsonl-out "$SAFELOOP_OUTPUT/decision_debug.jsonl"
```

Recommended collection principles:

- Sample regular trajectory windows for broad coverage.
- Sample more densely before hazards to improve early warning.
- Keep rollback pre/post windows so the decision head can learn whether intervention resolved the risk.
- Store raw images and JSONL files outside Git.

## Predictor Dataset

The multitask predictor uses recent visual observations, robot state, and the proposed next action. It predicts:

- future body or stuck hazard
- future body or stuck time-to-hazard
- future object hazard
- future object time-to-hazard
- current body or stuck hazard
- current object hazard

The future labels are read from the two-line assistant target in each JSONL sample. Current labels are either explicit in the sample or derived from marked trajectory steps with a configurable tolerance.

## Predictor Training

Train the frozen-backbone multitask head:

```bash
python scripts/train_qwen_multitask_safety.py \
  --model-dir "$QWEN_MODEL" \
  --data /path/to/merged_predictor_data.jsonl \
  --data-root /path/to/data_root \
  --out "$SAFELOOP_OUTPUT/qwen_multitask_head" \
  --epochs 1 \
  --batch-size 8 \
  --eval-batch-size 8 \
  --lr 6e-6 \
  --weight-decay 0.01 \
  --train-mode frozen \
  --bf16 \
  --device cuda \
  --current-tolerance 5 \
  --val-ratio 0.08 \
  --neck-size 512 \
  --dropout 0.12 \
  --max-grad-norm 1.0 \
  --future-bce-weight 1.05 \
  --future-tth-weight 1.0 \
  --current-bce-weight 1.55 \
  --body-weight 1.35 \
  --object-weight 1.20 \
  --current-positive-sample-weight 1.8 \
  --current-body-positive-sample-weight 6.0 \
  --current-object-positive-sample-weight 3.5 \
  --pos-weight-cap 30 \
  --save-steps 300
```

The head has a shared neck and three outputs: future risk logits, normalized time-to-hazard, and current-hazard logits.

## Decision-Head Training

Train the decision head online with closed-loop feedback:

```bash
python scripts/train_online_safeguard_decider.py \
  --policy-mode websocket \
  --benchmark libero_10 \
  --task-ids 0 3 4 \
  --predictor qwen-multitask \
  --qwen-model-path "$QWEN_MODEL" \
  --qwen-head-path /path/to/multitask_head.pt \
  --future-window 100 \
  --decision-interval 20 \
  --rollback-mode motion-plan \
  --motion-execution-mode kinematic \
  --allow-restore-fallback \
  --updates 6 \
  --episodes-per-update 3 \
  --ppo-epochs 4 \
  --lr 5e-5 \
  --bc-coef 0.50 \
  --completion-reward 10.0 \
  --body-hazard-penalty -0.35 \
  --object-hazard-penalty -0.18 \
  --stuck-hazard-penalty -0.75 \
  --rollback-penalty -0.02 \
  --rollback-failure-penalty -1.0 \
  --rollback-resolved-bonus 0.45 \
  --rollback-unresolved-penalty -0.45 \
  --record-penalty -0.0008 \
  --step-penalty -0.00002 \
  --out-dir "$SAFELOOP_OUTPUT/decision_head"
```

Reward design should prioritize task completion, penalize hazards, keep rollback costs nonzero, and reward rollback only when it resolves risk without destroying task progress.

## Compact Evaluation

Use the aggregator to compare compact summaries without committing raw traces:

```bash
python scripts/aggregate_closed_loop_results.py \
  --root baseline=/path/to/baseline_eval \
  --root safeloop=/path/to/safeloop_eval \
  --base-label baseline \
  --out-json "$SAFELOOP_OUTPUT/aggregate_report.json"
```

Commit only compact tables or summaries that are intended for public release. Keep detailed logs and videos as external artifacts.
