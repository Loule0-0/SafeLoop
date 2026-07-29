# SafeLoop Data and Training Pipeline

## Runtime

SafeLoop wraps a frozen manipulation policy with:

1. A short-horizon visual hazard predictor.
2. A memory of safe rollback anchors.
3. A decision head over `noop`, `record`, and `rollback`.
4. A motion-planning rollback executor.

At each decision interval, the predictor estimates current and future
body/stuck and object risk. The decision head can record the current simulator
state or select a recorded anchor for rollback. The frozen Pi0 policy resumes
after rollback and produces a new action chunk.

## Data Collection

```bash
python scripts/evaluate_pi0_safeguard_closed_loop.py \
  --libero-root "$LIBERO_ROOT" \
  --benchmark libero_10 \
  --task-id 0 \
  --episodes 1 \
  --policy-mode websocket \
  --predictor qwen-multitask \
  --qwen-model-path "$QWEN_MODEL" \
  --qwen-head-path "$SAFELOOP_WEIGHTS/prediction_heads/v23_success_balanced_step1000_multitask_head.pt" \
  --qwen-rollout-jsonl-out "$SAFELOOP_OUTPUT/qwen_rollout.jsonl" \
  --qwen-rollout-image-root "$SAFELOOP_OUTPUT/qwen_images" \
  --qwen-rollout-stride 30 \
  --qwen-rollout-prehazard-stride 8 \
  --decision-debug-jsonl-out "$SAFELOOP_OUTPUT/decision_debug.jsonl"
```

Each predictor sample contains recent camera frames, robot state, the proposed
action, future body/object hazard labels, normalized time-to-hazard targets,
and current body/object labels.

## Predictor Training

The predictor freezes Qwen2.5-VL-3B and trains a shared neck with future risk,
time-to-hazard, and current-risk outputs.

```bash
python scripts/run_release_predictor_training.py \
  --dataset-dir "$SAFELOOP_DATA" \
  --data-root "$SAFELOOP_DATA/materialized" \
  --model-dir "$QWEN_MODEL" \
  --weights-dir "$SAFELOOP_WEIGHTS" \
  --output-root "$SAFELOOP_OUTPUT"
```

The staged datasets, initialization heads, optimizer settings, and checkpoint
steps are defined in
[`configs/release/v56_predictor_training.json`](../configs/release/v56_predictor_training.json).

## Decision-Head Training

```bash
python scripts/run_release_decider_training.py \
  --model-dir "$QWEN_MODEL" \
  --weights-dir "$SAFELOOP_WEIGHTS" \
  --output-root "$SAFELOOP_OUTPUT" \
  --libero-root "$LIBERO_ROOT" \
  --openpi-root "$OPENPI_ROOT" \
  --checkpoint-dir "$PI0_CHECKPOINT_DIR"
```

The online trainer combines policy-gradient updates with behavior-cloning
regularization. Rollout features include predictor outputs, action statistics,
safe-anchor age and score, and previous intervention state.

## Evaluation

```bash
python scripts/run_release_24task_eval.py \
  --model-dir "$QWEN_MODEL" \
  --weights-dir "$SAFELOOP_WEIGHTS" \
  --output-root "$SAFELOOP_OUTPUT/eval" \
  --libero-root "$LIBERO_ROOT" \
  --openpi-root "$OPENPI_ROOT" \
  --checkpoint-dir "$PI0_CHECKPOINT_DIR"
```

The evaluator saves one summary, per-task summaries, episode records, and the
configured rollout videos under each output directory.
