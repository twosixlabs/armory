# CARLA Video Tracking Baseline Evaluation

This is the baseline evaluation for the single-object tracking scenario.  For multi-object tracking, see [carla_mot_results.md](../baseline_results/carla_mot_results.md).

For [dev data](https://github.com/twosixlabs/armory/blob/v0.15.2/armory/data/adversarial/carla_video_tracking_dev.py), results obtained using Armory v0.15.2.
For [test data](https://github.com/twosixlabs/armory/blob/v0.15.4/armory/data/adversarial/carla_video_tracking_test.py), results obtained using Armory v0.15.4.

| Data | Attack Parameters            | Benign Mean IoU | Benign Mean Success Rate | Adversarial Mean IoU | Adversarial Mean Success Rate | Test Size |
|------|------------------------------|-----------------|--------------------------|----------------------|-------------------------------|-----------|
| Dev  | step_size=0.02, max_iter=100 | 0.55/0.57       | 0.57/0.60                | 0.14/0.19            | 0.15/0.20                     | 20        |
| Test | step_size=0.02, max_iter=100 | 0.52/0.45       | 0.54/0.47                | 0.15/0.17            | 0.16/0.18                     | 20        |

a/b in the tables refer to undefended/defended performance results, respectively.

Find reference baseline configurations [here](https://github.com/twosixlabs/armory/tree/v0.15.4/scenario_configs/eval5/carla_video_tracking)