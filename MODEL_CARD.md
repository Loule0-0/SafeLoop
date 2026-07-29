# SafeLoop Model Card

## Architecture

SafeLoop is an outer-loop controller for frozen manipulation policies:

- Base policy: Pi0 LIBERO checkpoint.
- Predictor backbone: Qwen2.5-VL-3B-Instruct, frozen during released head
  training.
- Predictor head: future body/object risk, normalized time-to-hazard, and
  current body/object risk.
- Decision head: policy over `noop`, `record`, and `rollback`.
- Recovery: motion-planned execution to a selected safe anchor.

## Artifacts

- Weights: https://huggingface.co/Jaqen0-0/SafeLoop
- Training data: https://huggingface.co/datasets/Jaqen0-0/SafeLoop-Training-Data
- Weight revision: `pi0-v1`
- Weight checksums:
  `configs/release/artifacts_pi0_v1.json` in the GitHub repository.

The base Pi0 and Qwen checkpoints are downloaded from their upstream
repositories and retain their upstream licenses.

## Runtime

The release has been validated on Linux, Python 3.10, CUDA 12.4, and an NVIDIA
H20 GPU. The SafeLoop predictor and decision weights are loaded by
`scripts/quickstart_pi0_safeloop.sh`.

## License

The SafeLoop source code is licensed under Apache-2.0.
