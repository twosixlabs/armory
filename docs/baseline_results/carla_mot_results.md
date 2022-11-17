# CARLA MOT Baseline Evaluations

This is the baseline evaluation for the multi-object tracking scenario.  For single-object tracking, see [carla_video_tracking_results.md](../baseline_results/carla_video_tracking_results.md).

For [dev data](../armory/data/adversarial/carla_mot_dev.py), results obtained using Armory v0.16.1.

TODO: Add numbers
| Data | Defended | Attack            | Attack Parameters              | Benign DetA / AssA / HOTA | Adversarial DetA / AssA / HOTA | Test Size |
|------|---------------------------------------------------------------|---------------------------|--------------------------------|-----------|
| Dev  | no       | Adversarial Patch | step_size=0.02, max_iter=100   | 0.49 / 0.62 / 0.55        |  0.14 / 0.57 / 0.29            | 20        |
| Dev  | no       | Robust DPatch     | step_size=0.002, max_iter=1000 | X.XX / X.XX / X.XX        |  X.XX / X.XX / X.XX            | 20        |
| Dev  | yes      | Robust DPatch     | step_size=0.002, max_iter=1000 | X.XX / X.XX / X.XX        |  X.XX / X.XX / X.XX            | 20        |

Defended results not available for Adversarial Patch attack because JPEG Compression defense is not implemented in PyTorch and so is not fully differentiable.
Note that Robust DPatch is considerably slower than Adversarial Patch.

Find reference baseline configurations [here](../scenario_configs/eval6/carla_mot)