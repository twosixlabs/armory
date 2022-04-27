# Measurement Overview

Armory contains a number of functions to use as metrics as well as flexible measurement instrumentation.

For measuring and logging standard perturbation (e.g., `Lp` norms) and task metrics (e.g., `categorical_accuracy`) for model inputs and outputs, standard config usage will likely suffice.
For custom metrics and measuring intermediate values (e.g., outputs after a certain preprocessing layer), see the Instrumentation section below.

## Metrics

The `armory.utils.metrics` module implements functionality to measure both
task and perturbation metrics. 

### Config Usage


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

## Instrumentation

The `armory.instrument` module implements functionality to flexibly capture values for measurement.

The primary mechanisms are largely based off of the logging paradigm of loggers and handlers, though with significant differences on the handling side.

Probe - object to capture data and publish them for measurement.
Meter - object to measure a single metric with given captured data and output records
Writer - object to take meter output records and send them standard outputs (files, loggers, results dictionaries, etc.)
Hub - object to route captured probe data to meter inputs and route meter outputs to writers
There is typically only a single hub, where there can be numerous of the other types of objects.

### Quick Start

In order to capture and measure values, you need a Probe and a Meter connected to the hub, at a minimum:
```
from armory.instrument import get_probe, Meter, get_hub, PrintWriter
hub = get_hub()  # get global measurement hub
probe = get_probe("probe_name")  # get probe connected to global hub
meter = Meter("my_meter", lambda a,b: a+b, "probe_name.a", "probe_name.b")  # construct meter that measures the sum of a and b
hub.connect_meter(meter)  # connect meter to global hub

# # optionally, add a writer
writer = PrintWriter()
hub.connect_writer(writer, default=True)  # default sets all meters to use this writer

# Now, measure
probe.update(a=2, b=5)  # should also print to screen if PrintWriter is connected
probe.update(a=3)
probe.update(b=8)  # now it should print again
results = meter.results()
assert results == [7, 11]
```

Since these all use a global Hub object, it doesn't matter which python files they are instantatied in.
Probe should be instantiated in the file or class you are trying to measure.
Meters and writers can be instantiated in your initial setup, and can be connected before probes are constructed.

### Probes

To get a new Probe (connected to the default Hub):
```
# Module imports section
from armory.instrument import get_probe
probe = get_probe(name)
```
The arg `name` can be any `str` that is a valid python identifier, or can be blank, which defaults to the empty string `""`.
Similarly to `logging.getLogger`, this provides a namespace to place variables, and inputs like `__name__` can also be used.
Calls to `get_probe` using the same name will return the same Probe object.
The recommended approach is to set a probe at the top of the file of interest and use it for all captures in that file.

To capture values in-line, use `update`:
```
# ...
# In the code
probe.update(name=value)
```

This will publish the given value(s) to the given name(s) (also called probe variables) in the probe namespace of the connected Hub.
To be more concrete:
```
probe = get_probe("my.probe_name")
probe.update(arbitrary_variable_name=15)
```
will push the value 15 to `"my.probe_name.arbitrary_variable_name"`.
These names will be used when instantiating `Meter` objects.

However, this will fall on the floor (`del`, effectively) unless a meter is constructed and connected to the Hub to record values via `connect_meter`.
See the Quick Start section above or the Meters section below for more details.
This is analogous to having a `logging.Logger` without an appropriate `logging.Handler`.

Multiple variables can be updated simultaneously with a single function call (utilizing all kwargs given):
```
probe.update(a=x, b=y, c=z)
```

Sometimes it is helpful to perform preprocessing on the variables before publishing.
For instance, if the variable `y` was a pytorch tensor, it might be helpful to map to numpy via `y.detach().cpu().numpy()`.
However, it would be a waste of computation of nothing was set up to measure that value.
Therefore, probes leverage `args` to perform preprocessing on the input only when meters are connected.
For instance,
```
probe.update(lambda x: x.detach().cpu().numpy(), my_var=y)
```
Or, less succinctly,
```
probe.update(lambda x: x.detach(), lambda x: x.cpu(), lambda x: x.numpy(), my_var=y)
```
More generally,
```
probe.update(func1, func2, func3, my_var=y)
```
will publish the value `func3(func2(func1(y)))`. 

#### Hooking

Probes can also hook models to enable capturing values without modifying the target code.
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

#### Interactive Testing

An easy way to test probe outputs is to set the probe to a `MockSink` interface.
This can be done as follows:
```
from armory.instrument import get_probe, MockSink
probe = get_probe("my_name")
probe.set_sink(MockSink())
probe.update("
```
This will print all probe variables to the screen.

### Meter

A Meter is used to measure values output by probes.
It is essentially a wrapper around the functions of `armory.utils.metrics`.

To instantiate a Meter:
```
from armory.instrument import Meter
meter = Meter( 
        self,
        name,
        metric,
        *metric_arg_names,
        metric_kwargs=None,
        auto_measure=True,
        final=None,
        final_name=None,
        final_kwargs=None,
        keep_results=True,
)
```
It then needs to be connected to the hub:
```
from armory.instrument import get_hub
hub = get_hub()
hub.connect_meter(meter)


        StandardMeter(metrics.l2, "model.x_post[benign]", "model.x_post[adversarial]")

        probe_name.probe_variable[stage]

        metric_kwargs - kwargs that are constant across measurements
        auto_measure - whether to measure when all of the variables are present
            if False, 'measure()' must be called externally

        final - metric that takes in list of results as input
            Example: np.mean
        final_name - if final is not None, this is the name associated with the record
            if not specified, it defaults to f'{final}_{name}'

        keep_results - whether to locally store results
            if final is not None, keep_results is set to True


```
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

### Scenario usage

Make it clear that `scenario.x` refers to the `x` variable set in the scenario.

### Scenario config usage

TBD - I would like to get the machinery done separate from scenarios, then integrate.

To set one up (for scenarios) from a config, you can do:
```
# TBD
# instrument.MetricsMeter.from_config(config["metrics"])
```


