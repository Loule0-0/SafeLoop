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

## Not Included

This repository does not include base policy weights, Qwen weights, trained SafeLoop checkpoints, rollout images, videos, or raw training logs. Users must obtain compatible third-party assets separately and follow their licenses.

## Safety Notes

SafeLoop is intended for research use. Real-hardware deployment requires independent safety validation, conservative emergency stops, calibrated hazard definitions, and environment-specific testing.
