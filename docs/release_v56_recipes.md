# SafeLoop Pi0 Release Recipes

This page records the training and evaluation entry points for the `pi0-v1`
artifact set.

## Version Matrix

| Component | Version |
|---|---|
| SafeLoop package | `1.0.0` |
| SafeLoop weights | `Jaqen0-0/SafeLoop@pi0-v1` |
| Qwen backbone | `Qwen/Qwen2.5-VL-3B-Instruct@66285546d2b821cf421d4f5eb2576359d3770cd3` |
| Pi0 checkpoint | `gs://openpi-assets/checkpoints/pi0_libero` |
| LIBERO submodule | `8f1084e3132a39270c3a13ebe37270a43ece2a01` |
| OpenPI submodule | `b14bcf2989a46de9cc379f837b5a96a46a3948f4` |

The complete file sizes and SHA256 values for all released SafeLoop checkpoints
are stored in
[`configs/release/artifacts_pi0_v1.json`](../configs/release/artifacts_pi0_v1.json).

## Download Artifacts

```bash
python scripts/download_release_weights.py \
  --output-dir "$SAFELOOP_WEIGHTS"

hf download Jaqen0-0/SafeLoop-Training-Data \
  --repo-type dataset \
  --local-dir "$SAFELOOP_DATA"

python scripts/materialize_hf_training_data.py \
  --dataset-dir "$SAFELOOP_DATA" \
  --out "$SAFELOOP_DATA/materialized"
```

## Predictor Heads

The predictor recipe contains three stages:

- `v9_object_recall`
- `v23_success_balanced`
- `v30_lowdrift_success`

```bash
python scripts/run_release_predictor_training.py \
  --dataset-dir "$SAFELOOP_DATA" \
  --data-root "$SAFELOOP_DATA/materialized" \
  --model-dir "$QWEN_MODEL" \
  --weights-dir "$SAFELOOP_WEIGHTS" \
  --output-root "$SAFELOOP_OUTPUT"
```

Select one stage with `--recipe`, for example
`--recipe v30_lowdrift_success`.

## Decision Head

The decision head selects `noop`, `record`, or `rollback`.

```bash
python scripts/run_release_decider_training.py \
  --model-dir "$QWEN_MODEL" \
  --weights-dir "$SAFELOOP_WEIGHTS" \
  --output-root "$SAFELOOP_OUTPUT" \
  --libero-root "$LIBERO_ROOT" \
  --openpi-root "$OPENPI_ROOT" \
  --checkpoint-dir "$PI0_CHECKPOINT_DIR" \
  --policy-host 127.0.0.1 \
  --policy-port 8000
```

The recipe uses completion reward `+10.0`, body/stuck penalties `-2.0/-3.0`,
rollback failure `-9.0`, resolved rollback `+10.0`, unresolved rollback `-8.0`,
post-rollback episode success `+4.0`, and post-rollback episode failure `-5.5`.

## Evaluation

The configured task set is:

- LIBERO_OBJECT: `3, 9`
- LIBERO_GOAL: `0, 3, 5, 6, 9`
- LIBERO_10: `0, 3, 4, 6, 7, 9`
- LIBERO_SPATIAL: `0, 2, 4, 6`
- LIBERO_90: `0, 28, 30, 32, 37, 71, 84`

The suite horizons are Spatial `220`, Object `280`, Goal `300`, LIBERO-10
`520`, and LIBERO-90 `400` control steps.

```bash
python scripts/run_release_24task_eval.py \
  --model-dir "$QWEN_MODEL" \
  --weights-dir "$SAFELOOP_WEIGHTS" \
  --output-root "$SAFELOOP_OUTPUT/eval_24task" \
  --libero-root "$LIBERO_ROOT" \
  --openpi-root "$OPENPI_ROOT" \
  --checkpoint-dir "$PI0_CHECKPOINT_DIR" \
  --policy-host 127.0.0.1 \
  --policy-port 8000
```

Select a subset with `--task BENCHMARK:TASK_ID` and `--seeds`. The baseline
profile is `base_policy_reference`.
