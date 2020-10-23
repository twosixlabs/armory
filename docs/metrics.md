# Metrics

The `armory.utils.metrics` module implements functionality to measure both
task and perturbation metrics. 

### MetricsLogger

The `MetricsLogger` class pairs with scenarios to account for task performance
against benign and adversarial data as well as measure the perturbations of
adversarial samples. Since our datasets are presented as generators, this has
`update_task` and `update_perturbation` methods that can update metrics for
each batch obtained from the generator. The output, which is given by `results`,
is a JSON-able dict.

### Metrics

| Name | Type | Description |
|:-------: |:-------: |:-------: |
| categorical_accuracy | Task | Categorical Accuracy |
| object_detection_AP_per_class | Task | Average Precision @ IOU=0.5 |
| top_n_categorical_accuracy | Task | Top-n Categorical Accuracy |
| top_5_categorical_accuracy | Task | Top-5 Categorical Accuracy |
| word_error_rate | Task | Word Error Rate |
| image_circle_patch_diameter | Perturbation | Patch Diameter |
| lp   | Perturbation | L-p norm |
| linf | Perturbation | L-infinity norm |
| l2 | Perturbation | L2 norm |
| l1 | Perturbation | L1 norm |
| l0 | Perturbation | L0 "norm" |
| mars_mean_l2 | Perturbation | Mean L2 norm across video stacks |
| mars_mean_patch | Perturbation | Mean patch diameter across video stacks |
| norm | Perturbation | L-p norm |
| snr | Perturbation | Signal-to-noise ratio |
| snr_db | Perturbation | Signal-to-noise ratio (decibels) |
| snr_spectrogram | Perturbation | Signal-to-noise ratio of spectrogram |
| snr_spectrogram_db | Perturbation | Signal-to-noise ratio of spectrogram (decibels) |

<br>

We have implemented the metrics in numpy, instead of using framework-specific 
metrics, to prevent expanding the required set of dependencies.
