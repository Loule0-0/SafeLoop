# SafeLoop Model Card

Paper: SafeLoop: Risk-Aware Rollback for Vision-Language-Action Manipulation

## Overview

SafeLoop is an outer-loop safety controller for robotic manipulation research. It wraps a frozen base policy with a short-horizon hazard predictor, safe-anchor memory, motion-planning rollback, and a three-action decision head.

## Components

- Base policy: supplied by the user and kept frozen.
- Predictor: Qwen2.5-VL backbone with a lightweight multitask head.
- Decision head: policy over `noop`, `record`, and `rollback`.
- Rollback: motion-planning path execution to a selected safe anchor.

## Intended Use

- Simulation studies of safety wrappers for manipulation policies.
- LIBERO-style hazard prediction and rollback experiments.
- Offline and online training of safety prediction and decision modules.

## Public Artifacts

SafeLoop checkpoints and release training data are hosted outside Git:

- Weights: https://huggingface.co/Jaqen0-0/SafeLoop
- Training data: https://huggingface.co/datasets/Jaqen0-0/SafeLoop-Training-Data

The repository does not include base policy weights, Qwen weights, raw rollout traces, or raw training logs. Users must obtain compatible third-party assets separately and follow their licenses.

## Safety Notes

SafeLoop is intended for research use. Real-hardware deployment requires independent safety validation, conservative emergency stops, calibrated hazard definitions, and environment-specific testing.
