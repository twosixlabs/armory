# Measurement Overview

Armory contains a number of functions to use as metrics as well as flexible measurement instrumentation.

For measuring and logging standard perturbation (e.g., `Lp` norms) and task metrics (e.g., `categorical_accuracy`) for model inputs and outputs, standard config usage will likely suffice.
See the Metrics section for more information on available metrics.
For custom metrics and measuring intermediate values (e.g., outputs after a certain preprocessing layer), see the Instrumentation section below.

## Scenario Config Usage

In scenario configs, described in more detail in [Configuration File](docs/configuration_files.md), standard metrics can be added for measuring tasks and adversarial perturbations.
When running a scenario, these metrics are measured and output in json format in the results file.

Desired metrics and flags are placed under the key `"metric"` dictionary in the config:
```
"metric": {
    "means": [Bool],
    "perturbation": List[String] or String or null,
    "record_metric_per_sample": [Bool],
    "task": List[String] or String or null,
}
```
The `perturbation` and `task` fields can be null, a single string, or a list of strings.
Strings must be a valid armory metric from `armory.utils.metrics`, which are also described in the Metrics section below.
The perturbation metrics measure the difference between the benign and adversarial inputs `x`.
The task metrics measure the task performance on the predicted value w.r.t the true value `y`, for both benign and adversarial inputs.
These metrics are called on batches of inputs, but are sample-wise metrics, and so their results are concatenated to form a list over samples.

When `means` is true, the average value for the given metric is also recorded.
When `record_metric_per_sample` is true, all of the per-sample metrics are recorded.
If neither is true, a `ValueError` is raised, as nothing is recorded.

## Metrics

The `armory.utils.metrics` module implements functionality to measure both
task and perturbation metrics. 

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

- Probe - object to capture data and publish them for measurement.
- Meter - object to measure a single metric with given captured data and output records
- Writer - object to take meter output records and send them standard outputs (files, loggers, results dictionaries, etc.)
- Hub - object to route captured probe data to meter inputs and route meter outputs to writers
- There is typically only a single hub, where there can be numerous of the other types of objects.

### Quick Start

In order to capture and measure values, you need a Probe and a Meter connected to the hub, at a minimum:
```python
from armory.instrument import get_probe, Meter, get_hub, PrintWriter
hub = get_hub()  # get global measurement hub
probe = get_probe("probe_name")  # get probe connected to global hub
meter = Meter("my_meter", lambda a,b: a+b, "probe_name.a", "probe_name.b")  # construct meter that measures the sum of a and b
hub.connect_meter(meter)  # connect meter to global hub

# optionally, add a writer
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
```python
# Module imports section
from armory.instrument import get_probe
probe = get_probe(name)
```
The arg `name` can be any `str` that is a valid python identifier, or can be blank, which defaults to the empty string `""`.
Similarly to `logging.getLogger`, this provides a namespace to place variables, and inputs like `__name__` can also be used.
Calls to `get_probe` using the same name will return the same Probe object.
The recommended approach is to set a probe at the top of the file of interest and use it for all captures in that file.

To capture values in-line, use `update`:
```python
# ...
# In the code
probe.update(name=value)
```

This will publish the given value(s) to the given name(s) (also called probe variables) in the probe namespace of the connected Hub.
To be more concrete:
```python
probe = get_probe("my.probe_name")
probe.update(arbitrary_variable_name=15)
```
will push the value 15 to `"my.probe_name.arbitrary_variable_name"`.
These names will be used when instantiating `Meter` objects.

However, this will fall on the floor (`del`, effectively) unless a meter is constructed and connected to the Hub to record values via `connect_meter`.
See the Quick Start section above or the Meters section below for more details.
This is analogous to having a `logging.Logger` without an appropriate `logging.Handler`.

Multiple variables can be updated simultaneously with a single function call (utilizing all kwargs given):
```python
probe.update(a=x, b=y, c=z)
```

Sometimes it is helpful to perform preprocessing on the variables before publishing.
For instance, if the variable `y` was a pytorch tensor, it might be helpful to map to numpy via `y.detach().cpu().numpy()`.
However, it would be a waste of computation of nothing was set up to measure that value.
Therefore, probes leverage `args` to perform preprocessing on the input only when meters are connected.
For instance,
```python
probe.update(lambda x: x.detach().cpu().numpy(), my_var=y)
```
Or, less succinctly,
```python
probe.update(lambda x: x.detach(), lambda x: x.cpu(), lambda x: x.numpy(), my_var=y)
```
More generally,
```python
probe.update(func1, func2, func3, my_var=y)
```
will publish the value `func3(func2(func1(y)))`. 

#### Hooking

Probes can also hook models to enable capturing values without modifying the target code.
Currently, hooking is only implemented for PyTorch, but TensorFlow is on the roadmap.

To hook a model module, you can use the `hook` function.
For instance, 
```python
# probe.hook(module, *preprocessing, input=None, output=None)
probe.hook(convnet.layer1[0].conv2, lambda x: x.detach().cpu().numpy(), output="b")
```
This essentially wraps the `probe.update` call with a hooking function.
This is intended for usage that cannot or does not modify the target codebase.

More general hooking (e.g., for python methods) is TBD.

#### Interactive Testing

An easy way to test probe outputs is to set the probe to a `MockSink` interface.
This can be done as follows:
```python
from armory.instrument import get_probe, MockSink
probe = get_probe("my_name")
probe.set_sink(MockSink())
probe.update(variable_name=17)
# update probe variable my_name.variable_name to 17 
```
This will print all probe updates to the screen.

### Default Scenario Probe Values

The standard scenarios provide probe updates for the following variables:
- `i` - the current batch
- `x` - current batch of inputs
- `y` - current batch of ground truth labels
- `y_pred` - prediction of model on `x`
- `x_adv` - inputs perturbed by the current attack
- `y_pred_adv` - prediction of model on `x_adv`
- `y_target` (conditional) - target labels for attack, if attack is targeted

The standard probe used in scenarios is named `"scenario"`, so to access these, prepend the variable with `"scenario."`.
For example, the variable `x` set in the scenario can be referenced as `"scenario.x"`.

### Meter

A Meter is used to measure values output by probes.
It is essentially a wrapper around the functions of `armory.utils.metrics`, though it can employ any callable object.
You will need to construct a meter, connect it to a hub, and (optionally) add a writer.

#### Meter Construction

To instantiate a Meter:
```python
from armory.instrument import Meter
meter = Meter( 
    name,
    metric,
    *metric_arg_names,
    metric_kwargs=None,
    auto_measure=True,
    final=None,
    final_name=None,
    final_kwargs=None,
    record_final_only=False,
)
"""
A meter measures a function over specified input probe_variables for each update
    If final is not None, it also measures a function over those measurements
    Records are pushed to Writers for output

name - str name of meter, used when writing records
metric - callable function
metric_arg_names - str names of probe_variables corresponding to args passed into the metric function
    Meter(..., "model.x_post[benign]", "model.x_adv_post", ...)
    Follows the pattern of `probe_name.probe_variable[stage]` (stage is optional)
metric_kwargs - kwargs for the metric function that are constant across measurements

auto_measure - whether to measure when all of the variables have ben set
    if False, 'measure()' must be called externally

final - metric function that takes in the list of results as input (e.g., np.mean)
final_name - if final is not None, this is the name associated with the record
    if not specified, it defaults to f'{final}_{name}'
final_kwargs - kwargs for the final function that are constant across measurements
record_final_only - if True, do not record the standard metric, only final
    if record_final_only is True and final is None, no records are emitted
"""
```

For example, if you have a metric `diff`,
```python
def diff(a, b):
    return a - b
```
and you want to use it to measure the difference between `w` and `z` output from Probe `"my_probe"`, then you could do:
```python
meter = Meter(
    "my_meter_name",
    diff,
    "my_probe.w",
    "my_probe.z",
)
```
This will effectively call `diff(value["my_probe.w"], value["my_probe.z"])` once for each time both of those values are set.

If you wanted to take the average of diff over all the samples and only record that value, you would need to set final.
```python
meter = Meter(
    "name not recorded because record_final_only is True",
    diff,
    "my_probe.w",
    "my_probe.z",
    final=np.mean,
    final_name="mean_meter",  # actual recorded name
    final_kwargs=None,
    record_final_only=True,
)
``` 

The `metric_kwargs` and `final_kwargs` are a set of kwargs that are passed to each call of the corresponding function, but are assumed to be constant.
For example, this could be the `p` parameter in a generic `l_p` norm:
```python
def lp(x, x_adv, p=2):
    return np.linalg.norm(x-x_adv, ord=p, axis=1)

meter = Meter(
    "lp_perturbation",
    lp,
    "scenario.x",
    "scenario.x_adv",
    metric_kwargs={"p": 4},
)
```

#### Connecting Meter to Hub and Receiving Probe Updates

A constructed meter needs to be connected to a hub to receive `probe_variable` updates:
```python
from armory.instrument import get_hub
hub = get_hub()  # use global hub
hub.connect_meter(meter)
```

Updates are propagated to meters via the hub based on a simple filtering process.
If a probe named `probe_name` is updating a value `my_value` to 42, the call looks like this:
```python
get_probe("probe_name").update(my_value=42)
```
The hub then looks for a corresponding name from the lists of `metric_arg_names` from connected meters.
If the name is found, then the hub will call `set` on each of those meters, updating that argument value:
```python
meter.set("probe_name.my_value", 42, batch)
```
The `batch` arg is mainly used to track which iteration the meter is on, and is set automatically in scenarios.

Once all of the args have been set for a meter, it will call `self.measure()` if `auto_measure=True` (the default).
If `auto_measure=False`, then the user will need to explicitly call `meter.measure()`

NOTE: if `meter_arg_names` are misspelled, the meter will not measure anything.
This will log a warning if nothing has been called when meter.finalize() is called (typically via `hub.close()`), such as:
```python
Meter 'my_meter_name' was never measured. The following args were never set: ['probe_name.my_value']
```

#### Retrieving Results and Records

After measurement, the results are saved in a local list on the Meter and send records to any connected writers.
Similarly, after finalize is called, the final metric (if it is not `None`) will be applied to the results and saved in a local list, with a record sent to connected writers.

To retrieve a list of the values measured thus far, call `meter.results()`.
To retrieve the value computed by the final metric, call `meter.final_result()`.
If `measure` and `finalize` have not been called, respectively, then these will instead return `[]` and `None`.

Records are sent as 3-tuples to connected writers:
```python
(name, batch, result)
```
where `name` is the name given to the Meter, batch is the number set by the hub, and result is the result from calling the metric.
Final records are also 3-tuples:
```python
(final_name, None, final_result)
```
Note that the results stored by the meter are not the record tuples, but simply the raw results.

#### Connecting Writers

Armory scenarios will set up a default `ResultsWriter` that will take all connected meter records and write them to the output results json.
If additional outputs are desired, other Writer objects can be instantiated and connected to meters via the hub.

For instance, attaching a simple writer that prints all records to stdout:
```python
hub.connect_writer(PrintWriter(), default=True)
```

However, this can be quite verbose, so if you just want to add it to a particular meter, you can do this:
```python
hub.connect_meter(meter)  # meter must be connected before connecting a writer to it
hub.connect_writer(PrintWriter(), meters=[meter])
```

The are a number of different standard Writer objects:
- `Writer` - base class other writers are derived from
- `NullWriter` - writer that does nothing (writes to null)
- `LogWriter` - writer that writes to armory log in the given log level. Example: `LogWriter("WARNING")`
- `FileWriter` - writer that writes each record as a json-encoded line in the target file. Example: `FileWriter("records.txt")`
- `ResultsWriter` - writer that collates the records and outputs them as a dictionary. Used by scenarios as default.

To create a new Writer, simply subclass Writer and override the `_write` method (and optionally the `_close` method).

#### Stages and Update Filters

Conditional code may want to take advantage of being able to measure only at certain points in time or compare different stages of a scenario.

The `Hub` class contains context information that can be leveraged to filter out probe updates.
These are set by the `hub.set_context` method, and are automatically set by scenarios.
Currently, context contains the keys `batch` (number) and `stage`, which are respectively set to `int` and `str` values.
Future updates may extend the use of context information for more advanced filtering or measurement.

The batch number is incremented once per batch, is typically set to -1 prior to the first batch, and is primarily used internally by Meters to synchronize their measurements across stages.
The stage is intented primarily for filtering, starts with an empty string for a value, and is updated with logical parts of the scenario.
The primarily used scenario contexts (at present) for evasion attacks are:
- "next" - data iteration (get `x`, `y`, `i`, etc.)
- "benign" - model prediction on `x`
- "attack" - attack to generate `x_adv` from `x`
- "adversarial" - model prediction on `x_adv`
- "finished" - indicates that all benign and adversarial batches have been evaluated
Scenario contexts for poisoning scenarios are varied - see the scenarios for specifics.

We do not recommend directly setting context while running a scenario, or it will interfere with the standard meters.
However, these will likely need to be set when running a custom scenario and overriding standard methods like `next`, `run_benign`, and `run_attack`.

Probe updates can be filtered by meters by using a single bracketed string in the args list.
For instance, `"probe_name.probe_variable[adversarial]"` will only measure the value from `"probe_name.probe_variable"` when `stage = "adversarial"`.

This can be helpful when you want to measure something internal to a model but only during certain stages.
For instance, if you have a two stage model that applies preprocessing followed by estimation, and you want to measure the value after preprocessing:
```python
probe = get_probe("my_model")

def forward(self, x):
    x_proc = preprocessing(x)
    probe.update(x_after_preprocess=x_proc)
    y_pred = estimation(x_proc)
    return y_pred
```
You may want to compare the "linf" distance of `x_proc` in the benign case to `x_proc` for the corresponding adversarial case.
However, the model does not know the present context (whether it is being attacked or in otherwise), so measuring `"my_model.x_after_preprocess"` will get all of the forward passes caused by PGD.
In contrast, the following will directly measure the desired values:
```python
meter = Meter(
    "linf of x_proc benign vs adversarial",
    metrics.linf,
    "my_model.x_after_preprocess[benign]",
    "my_model.x_after_preprocess[adversarial]",
)
```
