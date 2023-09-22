# CARLA MOT Baseline Evaluations

This is the baseline evaluation for the multi-object tracking scenario.  For single-object tracking, see [carla_video_tracking_results.md](../baseline_results/carla_video_tracking_results.md).

Results obtained using Armory v0.18.0.



| Data | Defended | Attack            | Attack Parameters              | Benign DetA / AssA / HOTA | Adversarial DetA / AssA / HOTA | Test Size |
|------|----------|-------------------|--------------------------------|---------------------------|--------------------------------|-----------|
| Dev  | no       | Adversarial Patch | step_size=0.02, max_iter=100   | 0.55 / 0.64 / 0.59        |  0.15 / 0.58 / 0.29            | 20        |
| Dev  | no       | Robust DPatch     | step_size=0.002, max_iter=1000 | 0.55 / 0.64 / 0.59        |  0.42 / 0.61 / 0.50            | 20        |
| Dev  | yes      | Robust DPatch     | step_size=0.002, max_iter=1000 | 0.36 / 0.53 / 0.44        |  0.25 / 0.49 / 0.35            | 20        |
| Test  | no       | Adversarial Patch | step_size=0.02, max_iter=100   | 0.45 / 0.55 / 0.49        | 0.25 / 0.47 / 0.35             | 10        |
| Test  | no       | Robust DPatch     | step_size=0.002, max_iter=1000 | 0.45 / 0.55 / 0.49        | 0.36 / 0.49 / 0.41             | 10        |
| Test  | yes      | Robust DPatch     | step_size=0.002, max_iter=1000 | 0.31 / 0.44 / 0.37        | 0.22 / 0.39 / 0.29             | 10        |

Defended results not available for Adversarial Patch attack because JPEG Compression defense is not implemented in PyTorch and so is not fully differentiable.
Note that Robust DPatch is considerably slower than Adversarial Patch.

Find reference baseline configurations [here](https://github.com/twosixlabs/armory/tree/master/scenario_configs/eval7/carla_mot)