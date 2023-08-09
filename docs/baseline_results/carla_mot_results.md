# CARLA MOT Baseline Evaluations

This is the baseline evaluation for the multi-object tracking scenario.  For single-object tracking, see [carla_video_tracking_results.md](../baseline_results/carla_video_tracking_results.md).

For [dev data](https://github.com/twosixlabs/armory/blob/master/armory/data/adversarial/carla_mot_dev.py), results obtained using Armory v0.16.1.
[Test](https://github.com/twosixlabs/armory/blob/master/armory/data/adversarial/carla_mot_test.py) results obtained using Armory v0.17.2.


| Data | Defended | Attack            | Attack Parameters              | Benign DetA / AssA / HOTA | Adversarial DetA / AssA / HOTA | Test Size |
|------|----------|-------------------|--------------------------------|---------------------------|--------------------------------|-----------|
| Dev  | no       | Adversarial Patch | step_size=0.02, max_iter=100   | 0.49 / 0.62 / 0.55        |  0.18 / 0.57 / 0.32            | 20        |
| Dev  | no       | Robust DPatch     | step_size=0.002, max_iter=1000 | 0.49 / 0.62 / 0.55        |  0.39 / 0.59 / 0.48            | 20        |
| Dev  | yes      | Robust DPatch     | step_size=0.002, max_iter=1000 | 0.34 / 0.53 / 0.42        |  0.24 / 0.51 / 0.34            | 20        |
| Test  | no       | Adversarial Patch | step_size=0.02, max_iter=100   | 0.43 / 0.51 / 0.46        | 0.19 / 0.45 / 0.29             | 10        |
| Test  | no       | Robust DPatch     | step_size=0.002, max_iter=1000 | 0.43 / 0.51 / 0.46        | 0.31 / 0.46 / 0.37             | 10        |
| Test  | yes      | Robust DPatch     | step_size=0.002, max_iter=1000 | 0.32 / 0.45 / 0.38        | 0.22 / 0.41 / 0.30             | 10        |

Defended results not available for Adversarial Patch attack because JPEG Compression defense is not implemented in PyTorch and so is not fully differentiable.
Note that Robust DPatch is considerably slower than Adversarial Patch.

Find reference baseline configurations [here](https://github.com/twosixlabs/armory/tree/master/scenario_configs/eval7/carla_mot)