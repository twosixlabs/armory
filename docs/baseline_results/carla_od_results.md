# CARLA Object Detection Baseline Evaluations

## CARLA Street Level OD Dataset
(For [dev data](https://github.com/twosixlabs/armory/blob/v0.15.2/armory/data/adversarial/carla_obj_det_dev.py), results are obtained using Armory v0.15.2; for [test data](https://github.com/twosixlabs/armory/blob/v0.15.4/armory/data/adversarial/carla_obj_det_test.py), results are obtained using Armory v0.15.4)**

Single Modality (RGB) Object Detection

| Data | Attack  | Attack Parameters   | Benign  mAP | Benign  Disappearance  Rate | Benign  Hallucination  per Image | Benign  Misclassification  Rate | Benign  True Positive  Rate | Adversarial  mAP | Adversarial  Disappearance  Rate | Adversarial Hallucination  per Image | Adversarial Misclassification  Rate | Adversarial True Positive  Rate | Test Size |
|------|-------------------|------------------------------------|-------------|-----------------------------|----------------------------------|---------------------------------|-----------------------------|------------------|----------------------------------|--------------------------------------|-------------------------------------|---------------------------------|-----------|
| Dev  | Robust DPatch  | learning_rate=0.002, max_iter=2000 | 0.76/0.72   | 0.19/0.22 | 3.97/3.48 | 0.06/0.06  | 0.75/0.71 | 0.68/0.66   | 0.27/0.28 | 4.48/3.65  | 0.06/0.07    | 0.67/0.65  | 31   |
| Dev  | Adversarial Patch | learning_rate=0.003, max_iter=1000 | 0.76/0.72   | 0.19/0.22 | 3.97/3.48 | 0.06/0.06  | 0.75/0.71 | 0.54/*   | 0.32/*    | 22.16/*  | 0.05/*  | 0.62/*   | 31   |
| Test | Robust DPatch  | learning_rate=0.002, max_iter=2000 | 0.79/0.74   | 0.16/0.25 | 4.10/3.50 | 0.03/0.01  | 0.82/0.75 | 0.72/0.64   | 0.32/0.39 | 4.80/4.0   | 0.03/0.01    | 0.65/0.60  | 20   |
| Test | Adversarial Patch | learning_rate=0.003, max_iter=1000 | 0.79/0.74   | 0.16/0.25 | 4.10/3.50 | 0.03/0.01  | 0.82/0.75 | 0.38/*   | 0.40/*    | 42.55/*  | 0.03/*  | 0.57/*   | 20   |

Multimodality (RGB+depth) Object Detection

| Data | Attack  | Attack Parameters    | Benign  mAP | Benign  Disappearance  Rate | Benign  Hallucination  per Image | Benign  Misclassification  Rate | Benign  True Positive  Rate | Adversarial  mAP | Adversarial  Disappearance  Rate | Adversarial Hallucination  per Image | Adversarial Misclassification  Rate | Adversarial True Positive  Rate | Test Size |
|------|-------------------|--------------------------------------------------------------------------------------|-------------|-----------------------------|----------------------------------|---------------------------------|-----------------------------|------------------|----------------------------------|--------------------------------------|-------------------------------------|---------------------------------|-----------|
| Dev  | Robust DPatch  | depth_delta_meters=3, learning_rate=0.002, learning_rate_depth=0.0001, max_iter=2000 | 0.87/0.86   | 0.06/0.04 | 1.23/2.55 | 0.05/0.05  | 0.88/0.91 | 0.76/0.83   | 0.10/0.06 | 5.68/4.87  | 0.05/0.05    | 0.84/0.89  | 31   |
| Dev  | Adversarial Patch | depth_delta_meters=3, learning_rate=0.003, learning_rate_depth=0.0001, max_iter=1000 | 0.87/0.86   | 0.06/0.04 | 1.23/2.55 | 0.05/0.05  | 0.88/0.91 | 0.66/0.76   | 0.11/0.10 | 10.74/7.13    | 0.06/0.05    | 0.83/0.85  | 31   |
| Test | Robust DPatch  | depth_delta_meters=3, learning_rate=0.002, learning_rate_depth=0.0001, max_iter=2000 | 0.90/0.89   | 0.03/0.04 | 1.0/1.45  | 0.03/0.02  | 0.94/0.94 | 0.81/0.89   | 0.13/0.06 | 4.75/2.05  | 0.03/0.02    | 0.83/0.91  | 20   |
| Test | Adversarial Patch | depth_delta_meters=3, learning_rate=0.003, learning_rate_depth=0.0001, max_iter=1000 | 0.90/0.89   | 0.03/0.04 | 1.0/1.45  | 0.03/0.02  | 0.94/0.94 | 0.50/0.57   | 0.21/0.14 | 22.55/13.70   | 0.04/0.03    | 0.75/0.83  | 20   |

a/b in the tables refer to undefended/defended performance results, respectively.

\* Defended results not available for Adversarial Patch attack against single modality because JPEG Compression defense is not implemented in PyTorch and so is not fully differentiable

Find reference baseline configurations [here](https://github.com/twosixlabs/armory/tree/v0.15.4/scenario_configs/eval5/carla_object_detection)


## CARLA Overhead OD Dataset

Results obtained using Armory 0.18.1.

Single Modality (RGB) Object Detection

| Data | Split | Defended | Attack  | Attack Parameters   | Benign  mAP | Benign  Disappearance  Rate | Benign  Hallucination  per Image | Benign  Misclassification  Rate | Benign  True Positive  Rate | Adversarial  mAP | Adversarial  Disappearance  Rate | Adversarial Hallucination  per Image | Adversarial Misclassification  Rate | Adversarial True Positive  Rate | Test Size |
|------|---|----------|-------------------|------------------------------------|-------------|-----------------------------|----------------------------------|---------------------------------|-----------------------------|------------------|----------------------------------|--------------------------------------|-------------------------------------|---------------------------------|-----------|
| Dev | dev | no  | Adversarial Patch | learning_rate=0.05, max_iter=500, optimizer=Adam | 0.78   | 0.15 | 6.2  | 0.04  | 0.81 |  0.01  | 0.95   | 91.5  | 0.0   | 0.05  | 20   |
| Dev | dev | no | Adversarial Patch Targeted | learning_rate=0.05, max_iter=500, hallucination_per_label=300, optimizer=Adam | 0.78 | 0.15 | 6.2 | 0.04 | 0.81 | 0.44 | 0.42 | 67.2 | 0.03 | 0.55 | 20 |
| Dev | dev | no  | Robust DPatch  | learning_rate=0.002, max_iter=2000 | 0.78   | 0.15 | 6.2  | 0.04  | 0.81 |  0.69  | 0.24   | 7.85  | 0.03   | 0.72  | 20   |
| Dev | dev | yes | Robust DPatch  | learning_rate=0.002, max_iter=2000 | 0.62   | 0.37 | 3.0  | 0.03  | 0.60 |  0.50  | 0.46   | 9.4   | 0.03    | 0.51  | 20   |
| Test | test_hallucination | no | Robust DPatch  | learning_rate=0.002, max_iter=2000 | 0.74 | 0.15 | 3.6 | 0.05 | 0.80 | 0.32 | 0.18 | 30.3 | 0.04 | 0.78 | 25 |
| Test | test_disappearance | no | Robust DPatch  | learning_rate=0.002, max_iter=2000 | 0.74 | 0.25 | 5.36 | 0.03 | 0.72 | 0.63 | 0.34 | 8.12 | 0.02 | 0.64 | 25 |
| Test | test_hallucination | yes | Robust DPatch  | learning_rate=0.002, max_iter=2000 | 0.61 | 0.4 | 2.6 | 0.04 | 0.56 | 0.41 | 0.41 | 28.8 | 0.04 | 0.55 | 25 |
| Test | test_disappearance | yes | Robust DPatch  | learning_rate=0.002, max_iter=2000 | 0.56 | 0.46 | 3.7 | 0.02 | 0.52 | 0.42 | 0.55 | 14.5 | 0.01 | 0.44 | 25 |
| Test | test_hallucination | no | Adversarial Patch | learning_rate=0.05, max_iter=500, optimizer=Adam | 0.74 | 0.15 | 3.6 | 0.05 | 0.80 | 0.0 | 1.0 | 100.0 | 0.0 | 0.0 | 25 |
| Test | test_disappearance | no | Adversarial Patch | learning_rate=0.05, max_iter=500, optimizer=Adam | 0.74 | 0.25 | 5.36 | 0.03 | 0.72 | 0.01 | 0.98 | 99.2 | 0.0 | 0.02 | 25 |
| Test | test_hallucination | no | Adversarial Patch Targeted | learning_rate=0.05, max_iter=500, hallucination_per_label=300, optimizer=Adam | 0.74 | 0.15 | 3.6 | 0.05 | 0.80 | 0.60 | 0.28 | 71.2 | 0.04 | 0.68 | 25 |
| Test | test_disappearance | no | Adversarial Patch Targeted | learning_rate=0.05, max_iter=500, hallucination_per_label=300, optimizer=Adam | 0.74 | 0.25 | 5.4 | 0.03 | 0.72 | 0.44 | 0.48 | 64.6 | 0.02 | 0.50 | 25 |

Multimodality (RGB+depth) Object Detection

| Data | Split | Defended | Attack  | Attack Parameters    | Benign  mAP | Benign  Disappearance  Rate | Benign  Hallucination  per Image | Benign  Misclassification  Rate | Benign  True Positive  Rate | Adversarial  mAP | Adversarial  Disappearance  Rate | Adversarial Hallucination  per Image | Adversarial Misclassification  Rate | Adversarial True Positive  Rate | Test Size |
|------|---|----------|-------------------|-----------------------------------------------------------------------------------------|-------------|-----------------------------|----------------------------------|---------------------------------|-----------------------------|------------------|----------------------------------|--------------------------------------|-------------------------------------|---------------------------------|-----------|
| Dev | dev | no  | Adversarial Patch | depth_delta_meters=3, learning_rate=0.02, learning_rate_depth=0.0001, max_iter=1000, optimizer=Adam  | 0.79   | 0.13 | 4.5  | 0.04  | 0.83 | 0.18   | 0.38   | 39.0   | 0.03    | 0.59  | 20   |
| Dev | dev | yes | Adversarial Patch | depth_delta_meters=3, learning_rate=0.02, learning_rate_depth=0.0001, max_iter=1000, optimizer=Adam  | 0.80   | 0.14 | 2.8  | 0.03  | 0.83 | 0.21   | 0.39   | 31.2   | 0.02    | 0.59  | 20   |
| Dev | dev | no | Adversarial Patch Targeted | depth_delta_meters=3, learning_rate=0.02, learning_rate_depth=0.0001, max_iter=1000, optimizer=Adam, hallucination_per_label=300 | 0.79 | 0.13 | 4.5 | 0.04 | 0.83 | 0.67 | 0.21 | 17.6 | 0.05 | 0.74 | 20 | 
| Dev | dev | no  | Robust DPatch  | depth_delta_meters=3, learning_rate=0.002, learning_rate_depth=0.003, max_iter=2000 | 0.79   | 0.13 | 4.5  | 0.04  | 0.83 | 0.74   | 0.20   | 4.2  | 0.04    | 0.77  | 20   |
| Dev | dev | yes | Robust DPatch  | depth_delta_meters=3, learning_rate=0.002, learning_rate_depth=0.003, max_iter=2000 | 0.80   | 0.14 | 2.8  | 0.03  | 0.83 | 0.78   | 0.21   | 2.65   | 0.03    | 0.76  | 20   |
| Test | test_hallucination | no | Robust DPatch  | depth_delta_meters=3, learning_rate=0.002, learning_rate_depth=0.003, max_iter=2000 | 0.78 | 0.10 | 3.0 | 0.05 | 0.85 | 0.77 | 0.10 | 4.9 | 0.05 | 0.85 | 25 |
| Test | test_disappearance | no | Robust DPatch  | depth_delta_meters=3, learning_rate=0.002, learning_rate_depth=0.003, max_iter=2000 | 0.76 | 0.17 | 3.3 | 0.04 | 0.79 | 0.73 | 0.27 | 4.1 | 0.03 | 0.70 | 25 |
| Test | test_hallucination | yes | Robust DPatch  | depth_delta_meters=3, learning_rate=0.002, learning_rate_depth=0.003, max_iter=2000 | 0.82 | 0.10 | 1.96 | 0.05 | 0.84 | 0.81 | 0.11 | 2.08 | 0.05 | 0.83 | 25 |
| Test | test_disappearance | yes | Robust DPatch  | depth_delta_meters=3, learning_rate=0.002, learning_rate_depth=0.003, max_iter=2000 | 0.81 | 0.16 | 2.4 | 0.04 | 0.80 | 0.76 | 0.26 | 2.28 | 0.02 | 0.71 | 25 |
| Test | test_hallucination | no  | Adversarial Patch | depth_delta_meters=3, learning_rate=0.02, learning_rate_depth=0.0001, max_iter=1000, optimizer=Adam | 0.78 | 0.10 | 3.0 | 0.05 | 0.85 | 0.2 | 0.69 | 92.7 | 0.01 | 0.30 | 25 |
| Test | test_disappearance | no  | Adversarial Patch | depth_delta_meters=3, learning_rate=0.02, learning_rate_depth=0.0001, max_iter=1000, optimizer=Adam | 0.76 | 0.17 | 3.3 | 0.04 | 0.79 | 0.55 | 0.36 | 6.16 | 0.04 | 0.61 | 25 |
| Test | test_hallucination | yes | Adversarial Patch | depth_delta_meters=3, learning_rate=0.02, learning_rate_depth=0.0001, max_iter=1000, optimizer=Adam | 0.82 | 0.10 | 2.0 | 0.05 | 0.84 | 0.05 | 0.51 | 78.9 | 0.03 | 0.46 | 25 |
| Test | test_disappearance | yes | Adversarial Patch | learning_rate=0.02, learning_rate_depth=0.0001, max_iter=1000, optimizer=Adam | 0.81 | 0.16 | 2.4 | 0.04 | 0.80 | 0.45 | 0.36 | 12.3 | 0.03 | 0.62 | 25 |
| Test | test_hallucination | no | Adversarial Patch Targeted | depth_delta_meters=3, learning_rate=0.02, learning_rate_depth=0.0001, max_iter=1000, optimizer=Adam, hallucination_per_label=300 | 0.78 | 0.10 | 3.0 | 0.05 | 0.85 | 0.73 | 0.17 | 22.8 | 0.05 | 0.79 | 25 |
| Test | test_disappearance | no | Adversarial Patch Targeted | depth_delta_meters=3, learning_rate=0.02, learning_rate_depth=0.0001, max_iter=1000, optimizer=Adam, hallucination_per_label=300 | 0.76 | 0.17 | 3.28 | 0.04 | 0.79 | 0.69 | 0.27 | 15.0 | 0.03 | 0.70 | 25 |


Defended results not available for Adversarial Patch attack against single modality because JPEG Compression defense is not implemented in PyTorch and so is not fully differentiable

Find reference baseline configurations [here](https://github.com/twosixlabs/armory/tree/master/scenario_configs/eval7/carla_overhead_object_detection)