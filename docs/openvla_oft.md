# OpenVLA-OFT Backend

SafeLoop keeps the predictor unchanged and trains a policy-specific three-action decider from OpenVLA-OFT closed-loop rollouts. OpenVLA-OFT runs in a separate environment and exposes the same stateless websocket interface used by the LIBERO evaluator.

## Checkout and environment

```bash
git submodule update --init --recursive
CONDA_SH=/path/to/conda.sh bash scripts/setup_openvla_oft_env.sh
```

The OpenVLA-OFT submodule is pinned to the official implementation. Its official LIBERO recipe uses Python 3.10, PyTorch 2.2.0, the custom Transformers 4.40.1 fork, two images, proprioception, center crop, L1 action regression, and an eight-action chunk.

## Start the policy server

Run the server in the OpenVLA-OFT environment. `CUDA_VISIBLE_DEVICES` selects the GPU used by the 7B policy.

```bash
conda activate safeloop-openvla-oft
CUDA_VISIBLE_DEVICES=0 python scripts/serve_openvla_oft.py \
  --checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10 \
  --port 8001
```

The client sends the raw 256x256 LIBERO camera views after the required 180-degree flip. The server applies the official JPEG round trip, Lanczos resize, center crop, suite-specific proprio/action statistics, and LIBERO gripper conversion. A rollback clears the client action queue, so the next control step immediately requests a new OpenVLA-OFT action chunk.

The launcher disables the optional cuBLASLt `addmm` path for compatibility with H20 systems using the CUDA 12.1 runtime. Standard cuBLAS GEMM is used instead and preserves the model computation. Systems affected by a BF16 cuBLASLt kernel fault can additionally set `OPENVLA_INFERENCE_DTYPE=float16`; BF16 remains the default.

The combined public checkpoint has statistics for LIBERO-Spatial, Object, Goal, and LIBERO-10. It has no LIBERO-90 key. The release adapter therefore exposes `--libero-90-unnorm-key` and defaults to `libero_10_no_noops` for the paper's out-of-suite LIBERO-90 tasks.

## Train the OpenVLA-OFT decider

Run this command in the main SafeLoop environment while the policy server is active:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_release_decider_training.py \
  --config configs/release/openvla_oft_decider_training.json \
  --model-dir /path/to/Qwen2.5-VL-3B-Instruct \
  --weights-dir /path/to/SafeLoop/weights \
  --output-root outputs/openvla_oft \
  --policy-host 127.0.0.1 \
  --policy-port 8001
```

The recipe reuses `v23_success_balanced_step1000_multitask_head.pt` as the predictor, initializes from the released generic SafeLoop decider, and adapts only the 49D actor/56D asymmetric critic on LIBERO-10. It preserves the official eight-step OpenVLA-OFT action horizon and trains with the full 520-step LIBERO-10 episode limit, so PPO receives both completion rewards and failed-episode penalties instead of learning from short, success-free truncations.

For unattended runs, `scripts/launch_openvla_oft_decider_training.sh` records a status file and log under the selected output root. The release recipe uses a moderate object-hazard penalty, explicit record exploration, a two-rollback budget, and matching train/evaluation gates. Object rollback is allowed only from a high-confidence early prediction while current object contact is low. Rollback targets must be low-risk and 45-160 control steps old; an expired waypoint is never used as a fallback.

The OpenVLA-OFT recipe also enables a proprioceptive stuck fallback. It observes end-effector displacement over a 70-step window and promotes sustained low motion to the predictor's body-risk channel. This only opens the learned decider's rollback option and supplies an observable risk feature; it does not use simulator contacts or privileged hazard labels. The current-object veto is bypassed only after sustained stuck detection when the learned decider assigns rollback at least 0.95 probability, a safe waypoint exists, and the rollback budget permits it. Regular predictive rollback keeps the 160-step waypoint age cap; the sustained-stuck path may use a verified safe waypoint up to 320 steps old. A successful rollback refreshes the reached safe pose as a current anchor, so a later recovery returns to a recently verified state instead of an increasingly old trajectory point.

The launcher also supports staged continuation from an actor checkpoint. `UPDATE_START_INDEX` keeps checkpoint numbering and BC annealing on the original schedule, while `INIT_STATE_START` advances the LIBERO initial-state sequence. Place `OUTPUT_ROOT` on node-local scratch when the shared filesystem is congested, then copy the selected checkpoint to durable storage after evaluation.

```bash
OUTPUT_ROOT=/local_scratch/openvla_oft \
INITIAL_DECISION_CHECKPOINT=/path/to/online_decider_update002.pt \
UPDATE_START_INDEX=3 UPDATES=5 TOTAL_SCHEDULE_UPDATES=8 \
INIT_STATE_START=30 \
bash scripts/launch_openvla_oft_decider_training.sh
```

Actor-only continuation intentionally initializes a new critic and optimizer. It is a new PPO adaptation stage seeded by the selected deployed actor; `UPDATE_START_INDEX` preserves the actor's original checkpoint labels and behavior-cloning schedule rather than claiming an exact optimizer-state resume.

Before launching the full matrix, run a one-task evaluation that saves a video and marks hazards for manual review:

```bash
CUDA_VISIBLE_DEVICES=1 \
LIBERO_ROOT=/path/to/LIBERO \
QWEN_MODEL=/path/to/Qwen2.5-VL-3B-Instruct \
PREDICTOR_CHECKPOINT=/path/to/v23_success_balanced_step1000_multitask_head.pt \
DECISION_CHECKPOINT=/path/to/online_decider_update004.pt \
POLICY_PORT=8001 TASK_IDS=8 MODE=rl \
bash scripts/launch_openvla_oft_eval.sh
```

The launcher defaults to the suite's full 520-step limit, stores the summary, trace, and video in one run directory, and never treats automatic simulator proxies as paper hazard annotations.

The release deployment selects update 4 from the staged actor checkpoints. Its regular future-body gate is intentionally conservative (`probability >= 0.72` and `TTH <= 0.50`) to avoid interrupting normal low-speed manipulation. Proprioceptive stuck recovery remains independent: sustained low motion promotes the current body signal to an immediate hazard, so tightening the predictive gate does not disable stuck rollback.

For the full paper task matrix, use `scripts/run_release_24task_eval.py` with `configs/release/openvla_oft_24task_eval.json`. The default profile runs all 24 tasks and 16 paired seed/initial-state indices; `--seeds 0` runs a 24-task integration smoke test, and `--decision-checkpoint` selects an arbitrary candidate actor.
