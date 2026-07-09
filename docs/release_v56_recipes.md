# SafeLoop v56 Release Recipes

This page records the public training and evaluation entry points for the release associated with:

**SafeLoop: Risk-Aware Rollback for Vision-Language-Action Manipulation**

The repository keeps code and compact configs in Git. Large weights and training data are hosted separately:

- Weights: https://huggingface.co/Jaqen0-0/SafeLoop
- Training data: https://huggingface.co/datasets/Jaqen0-0/SafeLoop-Training-Data

## Download Artifacts

```bash
hf download Jaqen0-0/SafeLoop --local-dir $SAFELOOP_WEIGHTS
hf download Jaqen0-0/SafeLoop-Training-Data --repo-type dataset --local-dir $SAFELOOP_DATA
python scripts/materialize_hf_training_data.py \
  --dataset-dir $SAFELOOP_DATA \
  --out $SAFELOOP_DATA/materialized
```

The dataset includes three JSONL files and a de-duplicated image-frame shard set. The materialized image root should be passed as `--data-root`.

## Predictor Heads

The release uses a staged predictor-head family:

- `v9_object_recall`: object-recall head for object-heavy rollout states.
- `v23_success_balanced`: balanced head that preserves task success while improving current hazard separation.
- `v30_lowdrift_success`: low-drift stuck/object head that supplies the final step-300 and step-600 checkpoints.

Run the exact recipes:

```bash
python scripts/run_release_predictor_training.py \
  --dataset-dir $SAFELOOP_DATA \
  --data-root $SAFELOOP_DATA/materialized \
  --model-dir $QWEN_MODEL \
  --weights-dir $SAFELOOP_WEIGHTS \
  --output-root $SAFELOOP_OUTPUT
```

To run only one stage, add `--recipe v30_lowdrift_success`.

## Decision Head

The online decision head learns the three actions `noop`, `record`, and `rollback`. The release reward configuration is in `configs/release/v56_decider_training.json`.

```bash
python scripts/run_release_decider_training.py \
  --model-dir $QWEN_MODEL \
  --weights-dir $SAFELOOP_WEIGHTS \
  --output-root $SAFELOOP_OUTPUT \
  --libero-root $LIBERO_ROOT \
  --openpi-root $OPENPI_ROOT \
  --checkpoint-dir $PI0_CHECKPOINT_DIR \
  --policy-host 127.0.0.1 \
  --policy-port 8000
```

The core reward terms are completion `+10.0`, body/stuck penalties `-2.0/-3.0`, rollback failure `-9.0`, resolved rollback `+10.0`, unresolved rollback `-8.0`, episode success after rollback `+4.0`, and episode failure after rollback `-5.5`.

## 24-Task Evaluation

The 24-task suite is defined in `configs/release/v56_24task_eval.json`:

- LIBERO_OBJECT: `3, 9`
- LIBERO_GOAL: `0, 3, 5, 6, 9`
- LIBERO_10: `0, 3, 4, 6, 7, 9`
- LIBERO_SPATIAL: `0, 2, 4, 6`
- LIBERO_90: `0, 28, 30, 32, 37, 71, 84`

Run the default SafeLoop release matrix. This expands to 24 tasks x 16 seeds, for 384 rollouts.

```bash
python scripts/run_release_24task_eval.py \
  --model-dir $QWEN_MODEL \
  --weights-dir $SAFELOOP_WEIGHTS \
  --output-root $SAFELOOP_OUTPUT/eval_24task \
  --libero-root $LIBERO_ROOT \
  --openpi-root $OPENPI_ROOT \
  --checkpoint-dir $PI0_CHECKPOINT_DIR \
  --policy-host 127.0.0.1 \
  --policy-port 8000
```

The default profile is `safeloop_all`. The base-policy comparison uses `--profile pi0_baseline_reference`.

Paper hazard metrics are intended to be filled from manual video or trajectory review. The release profile enables `--manual-hazard-labels`, so automatic hazard counters in the JSON summaries are placeholders and should not be reported as paper safety numbers.
