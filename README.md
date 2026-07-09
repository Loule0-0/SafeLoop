# SafeLoop

SafeLoop is an outer-loop safety controller for frozen robotic manipulation policies. It monitors a base policy, records safe anchors, predicts short-horizon hazards, and can roll back to a safe anchor before handing control back to the base policy.

The repository focuses on three reproducible components:

- Collect closed-loop rollout data with hazard labels.
- Fine-tune a lightweight Qwen2.5-VL multitask prediction head.
- Train a three-action decision head for `noop`, `record`, and `rollback`.

Large artifacts such as model weights, rollout images, videos, checkpoints, and raw logs are intentionally excluded from the repository.

## Repository Layout

```text
safety_guard/
  controller.py                 SafeLoop controller and rollback interface
  memory.py                     safe-anchor memory
  libero_motion.py              LIBERO rollback motion planning
  libero_oracle.py              simulation hazard oracle
  qwen_multitask.py             multitask predictor dataset, head, loss, inference
  rl_policy_decider.py          three-action decision policy wrapper
  online_rl.py                  online policy optimization utilities
  rollout_sampling.py           rollout sampling for predictor data

scripts/
  evaluate_pi0_safeguard_closed_loop.py   closed-loop rollout and data collection
  build_qwen_hardcase_dataset.py          hard-case predictor dataset builder
  build_qwen_decision_rollout.py          decision rollout builder
  build_decision_rollout_from_pi0.py      offline decision rollout builder
  merge_decision_rollouts.py              decision dataset merger
  train_qwen_multitask_safety.py          predictor-head fine-tuning
  evaluate_qwen_multitask_safety.py       predictor-head evaluation
  train_online_safeguard_decider.py       online decision-head training
  train_three_action_decider.py           offline decision-head training
  aggregate_closed_loop_results.py        compact result aggregation
```

## Setup

Install the package in an environment with the required robotics, vision, and simulation dependencies:

```bash
git clone https://github.com/Loule0-0/SafeLoop.git
cd SafeLoop
pip install -e .
```

Set external asset paths through environment variables:

```bash
export LIBERO_ROOT=/path/to/LIBERO
export QWEN_MODEL=/path/to/Qwen2.5-VL-3B-Instruct
export PI0_POLICY=/path/to/pi0_libero_policy
export SAFELOOP_OUTPUT=/path/to/safeloop_outputs
```

The repository does not redistribute third-party model weights or datasets.

## Collect Predictor Data

Closed-loop rollout collection can save Qwen-style samples for future hazard prediction:

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
  --qwen-rollout-prehazard-stride 8
```

Each predictor sample contains recent camera frames, robot state, the proposed next action, and two future labels:

- body or stuck hazard probability and time-to-hazard
- object hazard probability and time-to-hazard

The dataset loader also derives current body/object hazard labels from marked trajectory steps when explicit current labels are unavailable.

## Train The Prediction Head

The default predictor training path freezes the Qwen2.5-VL backbone and trains a compact multitask head:

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

Evaluate the trained head with:

```bash
python scripts/evaluate_qwen_multitask_safety.py \
  --model-dir "$QWEN_MODEL" \
  --checkpoint /path/to/multitask_head.pt \
  --data /path/to/validation_data.jsonl \
  --data-root /path/to/data_root \
  --device cuda
```

## Train The Decision Head

The online decision trainer optimizes the three SafeLoop actions:

- `noop`: continue executing the base policy.
- `record`: save the current state as a rollback anchor.
- `rollback`: return to a selected safe anchor and resume execution.

Example:

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
  --rollback-resolved-bonus 0.45 \
  --rollback-unresolved-penalty -0.45 \
  --out-dir "$SAFELOOP_OUTPUT/decision_head"
```

## Artifact Policy

Keep these outside Git:

- base model weights
- trained checkpoints
- raw rollout images
- videos
- detailed traces and logs
- large JSONL datasets

For a paper release, publish compact tables in the paper or project page and provide external artifact links for weights and datasets when licenses allow.

## Tests

Run the unit tests with:

```bash
python -m unittest discover -s tests
```
