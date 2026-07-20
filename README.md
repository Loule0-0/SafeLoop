# SafeLoop: Risk-Aware Rollback for Vision-Language-Action Manipulation

---

## Overview

SafeLoop is an outer-loop safety framework for robotic manipulation policies. It detects risky trajectories before danger occurs, rolls back to a recorded safe state, and lets the policy replan from that state using its stochasticity and generalization ability.

This repository provides the complete code needed to:

- Collect closed-loop rollout data with hazard labels.
- Fine-tune a lightweight Qwen2.5-VL multitask prediction head.
- Train a three-action decision head for `noop`, `record`, and `rollback`.
- Run the same SafeLoop controller with Pi-0 or OpenVLA-OFT base policies.

Released weights and training data are hosted on Hugging Face:

- Weights: [Jaqen0-0/SafeLoop](https://huggingface.co/Jaqen0-0/SafeLoop)
- Training data: [Jaqen0-0/SafeLoop-Training-Data](https://huggingface.co/datasets/Jaqen0-0/SafeLoop-Training-Data)

The exact release training and 24-task evaluation recipes are documented in [docs/release_v56_recipes.md](docs/release_v56_recipes.md).

---

## Demo

The video below compares the same task and policy with and without SafeLoop.

![pi0 gets stuck while SafeLoop rolls back and completes the task](assets/demo/pi0_vs_safeloop_task09_seed214_ep0_labeled_red_green_boxes.gif)

Left: the original policy enters a stuck state during execution. Right: SafeLoop detects the risky trajectory, rolls back to a recorded safe waypoint without causing a hazard, and replans a successful path to finish the task. The original red and green overlays are retained for visual inspection.

---

## Repository Structure

```text
safety_guard/
  controller.py                 SafeLoop controller and rollback interface
  memory.py                     safe-anchor memory
  libero_motion.py              LIBERO rollback motion planning
  libero_oracle.py              optional simulation signals for data/debug
  qwen_multitask.py             multitask predictor dataset, head, loss, inference
  rl_policy_decider.py          three-action decision policy wrapper
  openvla_oft.py                OpenVLA-OFT normalization and action adapter
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
  run_release_predictor_training.py       v56 predictor training recipes
  run_release_decider_training.py         v56 decision-head training recipe
  run_release_24task_eval.py              v56 24-task evaluation runner
  materialize_hf_training_data.py         extract HF training-data shards
  check_release_environment.py            dependency, submodule, and artifact checks
  setup_release_env.sh                     reproducible Python environment setup
  serve_openvla_oft.py                     OpenVLA-OFT websocket policy server
  setup_openvla_oft_env.sh                 isolated OpenVLA-OFT environment
  aggregate_closed_loop_results.py        compact result aggregation

third_party/
  LIBERO/                                 pinned simulator submodule
  openpi/                                 pinned OpenPI policy/client submodule
  openvla-oft/                            pinned official OpenVLA-OFT submodule
```

---

## Installation

The tested release layout uses Python 3.10 for SafeLoop, Qwen, and LIBERO. The OpenPI and OpenVLA-OFT policy servers use separate environments so their model dependencies cannot conflict. LIBERO, OpenPI, and the official OpenVLA-OFT implementation are pinned as Git submodules.

### 1. Install System Dependencies

On Ubuntu, install the small set of system packages needed by Python virtual environments, video export, and headless MuJoCo rendering:

```bash
sudo apt-get update
sudo apt-get install -y python3.10 python3.10-venv git ffmpeg libegl1 libgl1 libglfw3 libosmesa6
```

The policy server also requires `uv`; follow the pinned OpenPI submodule's installation instructions if it is not already available.

### 2. Clone and Create the Environment

```bash
git clone https://github.com/Loule0-0/SafeLoop.git
cd SafeLoop
bash scripts/setup_release_env.sh
source .venv/bin/activate
```

The setup script initializes the two pinned top-level submodules. Do not install `third_party/LIBERO/requirements.txt` directly: that legacy file pins an old Transformers version that is incompatible with the Qwen2.5-VL predictor. The release setup installs the compatible simulation set from `requirements/libero-eval.txt` instead.

### 3. Configure Paths

Set external asset paths through environment variables:

```bash
export LIBERO_ROOT=/path/to/LIBERO
export OPENPI_ROOT=/path/to/openpi
export QWEN_MODEL=/path/to/Qwen2.5-VL-3B-Instruct
export PI0_CHECKPOINT_DIR=/path/to/pi0_libero_policy
export SAFELOOP_WEIGHTS=/path/to/safeloop_weights
export SAFELOOP_DATA=/path/to/safeloop_training_data
export SAFELOOP_OUTPUT=/path/to/safeloop_outputs
export MUJOCO_GL=egl
```

When using the bundled submodules, `LIBERO_ROOT` and `OPENPI_ROOT` can point to `$PWD/third_party/LIBERO` and `$PWD/third_party/openpi`.
On NVIDIA containers that omit the GLVND vendor registration, the rollout utilities automatically use `configs/runtime/10_nvidia.json` when `libEGL_nvidia.so.0` is available.

The repository does not redistribute third-party model weights or datasets.

### 4. Download Models and Data

```bash
hf download Jaqen0-0/SafeLoop --local-dir "$SAFELOOP_WEIGHTS"
hf download Jaqen0-0/SafeLoop-Training-Data --repo-type dataset --local-dir "$SAFELOOP_DATA"
hf download Qwen/Qwen2.5-VL-3B-Instruct --local-dir "$QWEN_MODEL"
python scripts/materialize_hf_training_data.py \
  --dataset-dir "$SAFELOOP_DATA" \
  --out "$SAFELOOP_DATA/materialized"
```

### 5. Validate the Installation

```bash
python scripts/check_release_environment.py --scope eval
```

---

## Start the Policy Server

SafeLoop's default release evaluation uses the OpenPI websocket client. Start the policy server in a separate terminal before launching data collection, training, or evaluation:

```bash
cd "$OPENPI_ROOT"
uv sync --frozen
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi0_libero \
  --policy.dir="$PI0_CHECKPOINT_DIR"
```

The server listens on `127.0.0.1:8000` by default. From the SafeLoop environment, verify both dependencies and connectivity with:

```bash
python scripts/check_release_environment.py \
  --scope eval \
  --check-policy-server \
  --policy-host 127.0.0.1 \
  --policy-port 8000
```

For OpenVLA-OFT, create its isolated environment and start the combined LIBERO checkpoint on a separate GPU:

```bash
CONDA_SH=/path/to/conda.sh bash scripts/setup_openvla_oft_env.sh
bash scripts/download_openvla_oft_checkpoint.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/launch_openvla_oft_server.sh
```

For unattended server setup, `scripts/launch_openvla_oft_download.sh` writes resumable download progress and status under `outputs/`.

The adapter uses the official two-view, proprioceptive, center-cropped, eight-action OFT inference path. See [docs/openvla_oft.md](docs/openvla_oft.md) for normalization-key handling and the complete training recipe.

Validate either policy endpoint with the same release checker:

```bash
python scripts/check_release_environment.py \
  --scope eval \
  --check-policy-server \
  --policy-backend openvla-oft \
  --policy-port 8001
```

---

## Collect Training Data

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

---

## Train the Prediction Head

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

For the exact release predictor recipe, use:

```bash
python scripts/run_release_predictor_training.py \
  --dataset-dir "$SAFELOOP_DATA" \
  --data-root "$SAFELOOP_DATA/materialized" \
  --model-dir "$QWEN_MODEL" \
  --weights-dir "$SAFELOOP_WEIGHTS" \
  --output-root "$SAFELOOP_OUTPUT"
```

---

## Train the Decision Head

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

For the release online decision-head recipe, use:

```bash
python scripts/run_release_decider_training.py \
  --model-dir "$QWEN_MODEL" \
  --weights-dir "$SAFELOOP_WEIGHTS" \
  --output-root "$SAFELOOP_OUTPUT" \
  --libero-root "$LIBERO_ROOT" \
  --openpi-root "$OPENPI_ROOT" \
  --checkpoint-dir "$PI0_CHECKPOINT_DIR"
```

To adapt the released predictor and generic decider to OpenVLA-OFT rollouts, keep the OpenVLA server active and run:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_release_decider_training.py \
  --config configs/release/openvla_oft_decider_training.json \
  --model-dir "$QWEN_MODEL" \
  --weights-dir "$SAFELOOP_WEIGHTS" \
  --output-root "$SAFELOOP_OUTPUT/openvla_oft" \
  --policy-port 8001
```

This recipe uses complete 400-step episodes from the seven paper LIBERO-90 tasks. It trains the policy-specific actor and asymmetric critic for two PPO updates with class-balanced behavior cloning and high rollback exploration. The selected `update001` actor is deployed behind stricter predictor gates and a recent-safe-anchor filter; the predictor weights remain unchanged. Following a rollback, the default OpenVLA-OFT deployment perturbs only the next arm-action chunk with a small, seed-reproducible, decaying offset so replanning can leave the failed trajectory while paired evaluations remain reproducible.

---

## Evaluation

Run the default 24-task SafeLoop evaluation matrix. This expands to 24 tasks x 16 seeds, matching the paper protocol.

```bash
python scripts/run_release_24task_eval.py \
  --model-dir "$QWEN_MODEL" \
  --weights-dir "$SAFELOOP_WEIGHTS" \
  --output-root "$SAFELOOP_OUTPUT/eval_24task" \
  --libero-root "$LIBERO_ROOT" \
  --openpi-root "$OPENPI_ROOT" \
  --checkpoint-dir "$PI0_CHECKPOINT_DIR"
```

The task list and seed list are stored in `configs/release/v56_24task_eval.json`. Each seed also selects the matching LIBERO initial-state index, so the default profile evaluates 16 seeds and 16 initial states per task. The default profile is `safeloop_all`; the baseline-only comparison is available as `--profile base_policy_reference`. Use `--seeds 0` for a 24-task smoke test without changing the release config.

The OpenVLA-OFT matrix keeps the same 24 tasks and 16 seeds while switching to its eight-action horizon and policy-specific decider. SafeLoop is enabled for every matrix entry, and each rollout is capped at its suite's full horizon. The public combined OpenVLA-OFT checkpoint natively provides statistics for the 17 selected Spatial, Object, Goal, and LIBERO-10 tasks. The seven LIBERO-90 entries are retained as explicitly out-of-suite tests through the documented normalization fallback and should be reported separately from native-suite results.

```bash
python scripts/run_release_24task_eval.py \
  --config configs/release/openvla_oft_24task_eval.json \
  --model-dir "$QWEN_MODEL" \
  --weights-dir "$SAFELOOP_WEIGHTS" \
  --output-root "$SAFELOOP_OUTPUT/openvla_oft_eval_24task" \
  --policy-port 8001
```

During checkpoint selection, pass `--decision-checkpoint /path/to/candidate.pt` to evaluate a candidate actor without editing the release config.

Use `scripts/launch_openvla_oft_eval.sh` for a one-task installation check before the full matrix. It runs the same SafeLoop gates, writes a status file, and saves the rollout video for manual hazard review.

Paper hazard metrics should be filled from manual video or trajectory review. The release runner therefore enables `--manual-hazard-labels` by default and marks automatic hazard fields as requiring review.

---

## Artifact Policy

Keep these outside Git:

- base model weights
- trained checkpoints stored directly in Git
- raw rollout images
- raw or large videos, except compact public demos under `assets/demo/`
- detailed traces and logs
- large JSONL datasets

For a paper release, publish compact tables in the paper or project page and provide external artifact links for weights and datasets when licenses allow.

---

## Tests

Run the unit tests with:

```bash
python -m unittest discover -s tests
```

---

## Citation

Citation information will be added after publication.
