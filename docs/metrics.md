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
|-------|-------|-------|
| categorical_accuracy | Task | Categorical Accuracy |
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
| object_detection_AP_per_class | Task | Object Detection mAP |
| object_detection_disappearance_rate | Task | Object Detection Disappearance Rate |
| object_detection_hallucinations_per_image| Task | Object Detection Hallucinations Per Image |
| object_detection_misclassification_rate | Task | Object Detection Misclassification Rate |
| object_detection_true_positive_rate | Task | Object Detection True Positive Rate | 
| snr | Perturbation | Signal-to-noise ratio |
| snr_db | Perturbation | Signal-to-noise ratio (decibels) |
| snr_spectrogram | Perturbation | Signal-to-noise ratio of spectrogram |
| snr_spectrogram_db | Perturbation | Signal-to-noise ratio of spectrogram (decibels) |

<br>

We have implemented the metrics in numpy, instead of using framework-specific 
metrics, to prevent expanding the required set of dependencies. Please see [armory/utils/metrics.py](../armory/utils/metrics.py) for more detailed descriptions.

### Targeted vs. Untargeted Attacks

For targeted attacks, each metric will be reported twice for adversarial data: once relative to the ground truth labels and once relative to the target labels.  For untargeted attacks, each metric is only reported relative to the ground truth labels.  Performance relative to ground truth measures the effectiveness of the defense, indicating the ability of the model to make correct predictions despite the perturbed input.  Performance relative to target labels measures the effectiveness of the attack, indicating the ability of the attacker to force the model to make predictions that are not only incorrect, but that align with the attackers chosen output.

# Instrumentation

The `armory.metrics.instrument` module implements functionality to flexibly capture values for measurement.

## Usage

The primary mechanisms are largely based off of the logging paradigm of loggers and handlers.
Loggers here are referred to as probes, and handlers are referred to as meters.

To capture values in-line, use probe:
```
# Module imports section
from armory.metrics import instrument
probe = instrument.get_probe()

# ...

# In the code
probe.update(name=value)
```

However, this will fall on the floor unless a meter is connected to record values.


### Preprocessing

Probes can perform preprocessing of the updating values.
This involves specifying a function or sequence of functions to perform when a value is updated.
However, this is done lazily - it is only performed if a meter is connected and is measuring.

For instance, to extract a pytorch tensor `layer_3_output`, you could do:
```
probe.update(lambda x: x.detach().cpu().numpy(), a=layer_3_output)
```
Or in a longer form:
```
probe.update(lambda x: x.detach(), lambda x: x.cpu().numpy(), a=layer_3_output)
```

This can also apply to multiple inputs:

```
probe.update(lambda x: x.detach().cpu().numpy(), a=layer_3_output, b=layer_4_output)
```

### Hooking

Currently, hooking is only implemented for PyTorch, but TensorFlow is on the roadmap.

To hook a model module, you can use the `hook` function.
For instance, 
```
# probe.hook(module, *preprocessing, input=None, output=None)
probe.hook(convnet.layer1[0].conv2, lambda x: x.detach().cpu().numpy(), output="b")
```

This essentially wraps the `probe.update` call with a hooking function.
This is intended for usage that cannot or does not modify the target codebase.

More general hooking (e.g., for python methods) is TBD.

## Meter

To set up a meter, you need to connect it to the probe.
This can be done as follows:
```
from armory.metrics import instrument
meter = instrument.LogMeter()
probe.connect(meter)
```

When a probe value is updated, it first calls `is_measuring`.
If `False`, nothing is done.
If `True`, preprocessing is run on the probe data and `meter.update` is called with it.

The default `NullMeter` and `LogMeter` classes always return `True` when `is_measuring` is called.
In custom Meters, it may be desirable to override this function to reduce computational cost.

### Update and Measure

When the probes update and it passes the `is_measuring` check, the `update` method is called.
This can be used to directly process the values as they change.

Alternatively, the `measure` method can be used to process the current set of values.
This can be helpful when the calling program has more control over the executing code.

### Stages and Steps

Conditional code may want to take advantage of being able to measure only at certain points in time or compare different stages of a scenario.

Probes are deliberately "dumb" - they do not have knowledge or context, beyond their construction.
This is in contrast to TensorBoard, which requires steps to be directly input in-line.
The TensorBoard approach can be very difficult to use when that context is unavailable or hard to provide.

An example is probing an intermediate model value and comparing benign and adversarial activations.
Since the model doesn't know whether it is being attacked, it would require passing that in from outside.

The `Meter` class has `set_stage` and `set_step` methods for setting where an experiment is at currently.
These can then be used to provide more granular measurement.

Here is an example meter:
```
class DiffMeter(Meter):
    def is_measuring(self, name):
        return (name == "z" and self.stage in ("run_benign", "predict_attack"))

    def update(self, z=None):
        if self.stage == "run_benign":
            self.z = z
        elif self.stage == "predict_attack":
            self.z_adv = z
            diff = np.abs((z - z_adv)).sum()
            log.info(f"diff is {diff}")

```

David's note:
Thinking about this more, I think having a third class, like "Bench" or "Experiment" that keeps track of steps and stages would be helpful.
Where we can have many-to-many relationships between meters and probes, we could have a single bench that keeps track of that sort of overall state.

### Scenario config usage

TBD - I would like to get the machinery done separate from scenarios, then integrate.

To set one up (for scenarios) from a config, you can do:
```
# TBD
# instrument.MetricsMeter.from_config(config["metrics"])
```


