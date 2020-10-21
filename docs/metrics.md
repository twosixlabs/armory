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
| top_5_categorical_accuracy | Task | Top-5 Categorical Accuracy |
| object_detection_AP_per_class | Task | Average Precision @ IOU=0.5 |
| linf | Perturbation | L-infinity norm |
| l2 | Perturbation | L2 norm |
| l1 | Perturbation | L1 norm |
| l0 | Perturbation | L0 "norm" |
| image_circle_patch_diameter | Perturbation | Patch Diameter |

<br>

We have implemented the metrics in numpy, instead of using framework-specific 
metrics, to prevent expanding the required set of dependencies.
