# CARLA Object Detection Baseline Evaluations

## CARLA Street Level OD Dataset
(For [dev data](https://github.com/twosixlabs/armory/blob/v0.15.2/armory/data/adversarial/carla_obj_det_dev.py), results are obtained using Armory v0.15.2; for [test data](https://github.com/twosixlabs/armory/blob/v0.15.4/armory/data/adversarial/carla_obj_det_test.py), results are obtained using Armory v0.15.4)**

Single Modality (RGB) Object Detection
| Data | Attack            | Attack Parameters                  | Benign  mAP | Benign  Disappearance  Rate | Benign  Hallucination  per Image | Benign  Misclassification  Rate | Benign  True Positive  Rate | Adversarial  mAP | Adversarial  Disappearance  Rate | Adversarial Hallucination  per Image | Adversarial Misclassification  Rate | Adversarial True Positive  Rate | Test Size |
|------|-------------------|------------------------------------|-------------|-----------------------------|----------------------------------|---------------------------------|-----------------------------|------------------|----------------------------------|--------------------------------------|-------------------------------------|---------------------------------|-----------|
| Dev  | Robust DPatch     | learning_rate=0.002, max_iter=2000 | 0.76/0.72   | 0.19/0.22                   | 3.97/3.48                        | 0.06/0.06                       | 0.75/0.71                   | 0.68/0.66        | 0.27/0.28                        | 4.48/3.65                            | 0.06/0.07                           | 0.67/0.65                       | 31        |
| Dev  | Adversarial Patch | learning_rate=0.003, max_iter=1000 | 0.76/0.72   | 0.19/0.22                   | 3.97/3.48                        | 0.06/0.06                       | 0.75/0.71                   | 0.54/*           | 0.32/*                           | 22.16/*                              | 0.05/*                              | 0.62/*                          | 31        |
| Test | Robust DPatch     | learning_rate=0.002, max_iter=2000 | 0.79/0.74   | 0.16/0.25                   | 4.10/3.50                        | 0.03/0.01                       | 0.82/0.75                   | 0.72/0.64        | 0.32/0.39                        | 4.80/4.0                             | 0.03/0.01                           | 0.65/0.60                       | 20        |
| Test | Adversarial Patch | learning_rate=0.003, max_iter=1000 | 0.79/0.74   | 0.16/0.25                   | 4.10/3.50                        | 0.03/0.01                       | 0.82/0.75                   | 0.38/*           | 0.40/*                           | 42.55/*                              | 0.03/*                              | 0.57/*                          | 20        |

Multimodality (RGB+depth) Object Detection
| Data | Attack            | Attack Parameters                                                                    | Benign  mAP | Benign  Disappearance  Rate | Benign  Hallucination  per Image | Benign  Misclassification  Rate | Benign  True Positive  Rate | Adversarial  mAP | Adversarial  Disappearance  Rate | Adversarial Hallucination  per Image | Adversarial Misclassification  Rate | Adversarial True Positive  Rate | Test Size |
|------|-------------------|--------------------------------------------------------------------------------------|-------------|-----------------------------|----------------------------------|---------------------------------|-----------------------------|------------------|----------------------------------|--------------------------------------|-------------------------------------|---------------------------------|-----------|
| Dev  | Robust DPatch     | depth_delta_meters=3, learning_rate=0.002, learning_rate_depth=0.0001, max_iter=2000 | 0.87/0.86   | 0.06/0.04                   | 1.23/2.55                        | 0.05/0.05                       | 0.88/0.91                   | 0.76/0.83        | 0.10/0.06                        | 5.68/4.87                            | 0.05/0.05                           | 0.84/0.89                       | 31        |
| Dev  | Adversarial Patch | depth_delta_meters=3, learning_rate=0.003, learning_rate_depth=0.0001, max_iter=1000 | 0.87/0.86   | 0.06/0.04                   | 1.23/2.55                        | 0.05/0.05                       | 0.88/0.91                   | 0.66/0.76        | 0.11/0.10                        | 10.74/7.13                           | 0.06/0.05                           | 0.83/0.85                       | 31        |
| Test | Robust DPatch     | depth_delta_meters=3, learning_rate=0.002, learning_rate_depth=0.0001, max_iter=2000 | 0.90/0.89   | 0.03/0.04                   | 1.0/1.45                         | 0.03/0.02                       | 0.94/0.94                   | 0.81/0.89        | 0.13/0.06                        | 4.75/2.05                            | 0.03/0.02                           | 0.83/0.91                       | 20        |
| Test | Adversarial Patch | depth_delta_meters=3, learning_rate=0.003, learning_rate_depth=0.0001, max_iter=1000 | 0.90/0.89   | 0.03/0.04                   | 1.0/1.45                         | 0.03/0.02                       | 0.94/0.94                   | 0.50/0.57        | 0.21/0.14                        | 22.55/13.70                          | 0.04/0.03                           | 0.75/0.83                       | 20        |

a/b in the tables refer to undefended/defended performance results, respectively.

\* Undefended results not available for Adversarial Patch attack against single modality because JPEG Compression defense is not implemented in PyTorch and so is not fully differentiable

Find reference baseline configurations [here](https://github.com/twosixlabs/armory/tree/v0.15.4/scenario_configs/eval5/carla_object_detection)


## CARLA Overhead OD Dataset

TODO: add numbers

Single Modality (RGB) Object Detection
| Data | Attack            | Attack Parameters                  | Benign  mAP | Benign  Disappearance  Rate | Benign  Hallucination  per Image | Benign  Misclassification  Rate | Benign  True Positive  Rate | Adversarial  mAP | Adversarial  Disappearance  Rate | Adversarial Hallucination  per Image | Adversarial Misclassification  Rate | Adversarial True Positive  Rate | Test Size |
|------|-------------------|------------------------------------|-------------|-----------------------------|----------------------------------|---------------------------------|-----------------------------|------------------|----------------------------------|--------------------------------------|-------------------------------------|---------------------------------|-----------|



Multimodality (RGB+depth) Object Detection
| Data | Attack            | Attack Parameters                                                                    | Benign  mAP | Benign  Disappearance  Rate | Benign  Hallucination  per Image | Benign  Misclassification  Rate | Benign  True Positive  Rate | Adversarial  mAP | Adversarial  Disappearance  Rate | Adversarial Hallucination  per Image | Adversarial Misclassification  Rate | Adversarial True Positive  Rate | Test Size |
|------|-------------------|--------------------------------------------------------------------------------------|-------------|-----------------------------|----------------------------------|---------------------------------|-----------------------------|------------------|----------------------------------|--------------------------------------|-------------------------------------|---------------------------------|-----------|