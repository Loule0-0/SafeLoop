# Demo Assets

This directory contains compact, manually reviewed SafeLoop demos. A demo may come from any LIBERO task; it does not need to belong to the paper's 24-task evaluation suite. Automatic simulator hazard counters are not used to certify these examples.

| Files | Task | Outcome |
|---|---|---|
| `pi0_vs_safeloop_task09_seed214_ep0_labeled_red_green_boxes.{gif,mp4}` | LIBERO_10 task 9, seed 214 | pi0 becomes stuck; SafeLoop rolls back and completes the same task. |
| `safeloop_libero10_task06_seed389_rollback_success.{gif,mp4}` | LIBERO_10 task 6, seed 389 | SafeLoop rolls back at step 50, avoids contact during recovery, and completes the task. |

The side-by-side demo keeps the original red/green overlays. In the SafeLoop-only demo, the green border denotes SafeLoop execution and the orange `ROLLBACK` label marks the recovery segment.

Only rollouts that visibly demonstrate rollback without a resulting hazard are retained. Failed or visually unsafe candidates are excluded.
