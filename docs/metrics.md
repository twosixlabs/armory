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
    "max_record_size": Integer or null,
    "means": [Bool],
    "perturbation": List[String] or String or null,
    "profiler_type": null or "basic" or "deterministic",
    "record_metric_per_sample": [Bool],
    "task": List[String] or String or null,
    "task_kwargs": List[dict] (Optional)
}
```
The `perturbation` and `task` fields can be null, a single string, or a list of strings.
The perturbation metrics measure the difference between the benign and adversarial inputs `x`.
The task metrics measure the task performance on the predicted value w.r.t the true value `y`, for both benign and adversarial inputs.
If task metrics take keyword arguments, such as `"iou_threshold"`, these can be (optionally) added a list of kwarg dicts.
The `task_kwargs` list must be the same length as `task`, and metrics without kwargs should have `{}` as a placeholder.
These metrics are called on batches of inputs, but are sample-wise metrics, and so their results are concatenated to form a list over samples.

When `means` is true, the average value for the given metric is also recorded.
When `record_metric_per_sample` is true, all of the per-sample metrics are recorded.
If neither is true, a `ValueError` is raised, as nothing is recorded.
The `max_record_size` field, if not `null`, will drop individual records sent to the ResultsWriter that are greater than the given value.
    To use the default of `2**20` bytes (per record, not per full results output), do not include this field in the config.

The `profiler_type` field, when not `null`, enables the logging of computational metrics.
If `"basic"`, it logs CPU time for model inference and attacking.
If `"deterministic"`, which runs *very* slowly, also provides verbose CPU statistics at the function call level, like so:
```
         837 function calls (723 primitive calls) in 0.063 seconds

   Ordered by: cumulative time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.063    0.063 /opt/conda/lib/python3.8/site-packages/art/attacks/evasion/fast_gradient.py:207(generate)
        1    0.000    0.000    0.054    0.054 /opt/conda/lib/python3.8/site-packages/art/attacks/evasion/fast_gradient.py:477(_compute)
        1    0.000    0.000    0.053    0.053 /opt/conda/lib/python3.8/site-packages/art/attacks/evasion/fast_gradient.py:383(_compute_perturbation)
        1    0.000    0.000    0.052    0.052 /opt/conda/lib/python3.8/site-packages/art/estimators/classification/keras.py:422(loss_gradient)
        1    0.000    0.000    0.052    0.052 /opt/conda/lib/python3.8/site-packages/keras/backend.py:4238(__call__)
        1    0.000    0.000    0.042    0.042 /opt/conda/lib/python3.8/site-packages/keras/backend.py:4170(_make_callable)
        1    0.000    0.000    0.042    0.042 /opt/conda/lib/python3.8/site-packages/tensorflow/python/client/session.py:1502(_make_callable_from_options)
   ...
```
Profiler information can be found in the results json under `["results"]["compute"]`.
The functionality for these profilers can be found in `armory/metrics/compute.py`.

### Targeted vs. Untargeted Attacks

For targeted attacks, each metric will be reported twice for adversarial data: once relative to the ground truth labels and once relative to the target labels.  For untargeted attacks, each metric is only reported relative to the ground truth labels.  Performance relative to ground truth measures the effectiveness of the defense, indicating the ability of the model to make correct predictions despite the perturbed input.  Performance relative to target labels measures the effectiveness of the attack, indicating the ability of the attacker to force the model to make predictions that are not only incorrect, but that align with the attackers chosen output.

## Metrics

The `armory.metrics` module contains functionality to measure a variety of metrics:
- `armory.metrics.perturbation` metrics measure adversarial perturbations such as `lp` distance
- `armory.metrics.task` metrics measure task performance such as categorical accuracy 
- `armory.metrics.statistical` metrics measure statistical quantities such as KL divergence
- `armory.metrics.poisoning` module contains helper functions to measure fairness statistics relevant to poisoning and filtering scenarios

We have implemented the metrics in numpy, instead of using framework-specific metrics, to prevent expanding the required set of dependencies.
Please see the relevant submodules in [armory/metrics](../armory/metrics/) for more detailed descriptions.

### Perturbation Metrics

Perturbation metrics compare a benign and adversarially perturbed input and return a distance.
Typically, these functions follow the form of `func(x, x_adv)`, where `x` is the benign input and `x_adv` is the perturbed input.

The set of perturbation metrics provided by armory can also be via batch-wise and element-wise namespaces as follows:
```python
from armory.metrics import perturbation
print(peturbation.batch)
# ['image_circle_patch_diameter', 'l0', 'l1', 'l2', 'linf', 'max_image_circle_patch_diameter', 'max_l0', 'max_l1', 'max_l2', 'max_linf', 'mean_image_circle_patch_diameter', 'mean_l0', 'mean_l1', 'mean_l2', 'mean_linf', 'snr', 'snr_db', 'snr_spectrogram', 'snr_spectrogram_db']
print(perturbation.element)
# ['image_circle_patch_diameter', 'l0', 'l1', 'l2', 'linf', 'max_image_circle_patch_diameter', 'max_l0', 'max_l1', 'max_l2', 'max_linf', 'mean_image_circle_patch_diameter', 'mean_l0', 'mean_l1', 'mean_l2', 'mean_linf', 'snr', 'snr_db', 'snr_spectrogram', 'snr_spectrogram_db']
```
Currently, all perturbation metrics have element-wise and batch-wise versions, though the config assumes that the batch version is intended.
For instance:
```python
perturbation.batch.l1([0, 0, 0], [1, 1, 1])
# array([1., 1., 1.])
perturbation.element.l1([0, 0, 0], [1, 1, 1])
# 3.0
```
Metric outputs are numpy arrays or scalars.

| Name | Namespace | Description |
|-------|-------|-------|
| `linf` | `perturbation.batch.linf` | L-infinity norm |
| `l2` | `perturbation.batch.l2` | L2 norm |
| `l1` | `perturbation.batch.l1` | L1 norm |
| `l0` | `perturbation.batch.l0` | L0 "norm" |
| `snr` | `perturbation.batch.snr` | Signal-to-noise ratio |
| `snr_db` | `perturbation.batch.snr_db` | Signal-to-noise ratio (decibels) |
| `snr_spectrogram` | `perturbation.batch.snr_spectrogram` | Signal-to-noise ratio of spectrogram |
| `snr_spectrogram_db` | `perturbation.batch.snr_spectrogram_db` | Signal-to-noise ratio of spectrogram (decibels) |
| `image_circle_patch_diameter` | `perturbation.batch.image_circle_patch_diameter` | Diameter of smallest circular patch |
| `mean_l(0\|1\|2\|inf)` | `perturbation.batch.mean_l(0\|1\|2\|inf)` | Lp norm averaged over all frames of video |
| `max_l(0\|1\|2\|inf)` | `perturbation.batch.max_l(0\|1\|2\|inf)` | Max of Lp norm over all frames of video |
| `(mean\|max)_image_circle_patch_diameter` | `perturbation.batch.(mean\|max)` | Average or max circle over all frames of video |

<br>

### Task Metrics

The `metrics.task` module contains metrics for measurement of task performance.
Generally, these functions follow the form of `func(y, y_pred)`, where `y` is the ground truth and `y_pred` is the prediction.
This is true for all of the batchwise and elementwise functions (which behave similarly to the `perturbation` module).
However, not all batchwise functions have elementwise counterparts (e.g., `per_class_accuracy`).

Those metrics in the `population` namespace take `y_list` and `y_pred_list`, which are indicative of the entire dataset.
They can be called on a subset of the population, but for a correct overall result, it requires the entire set of predictions.

Some metrics such as total word error rate and mean average precision are effectively aggregations of batchwise metrics, and are in the `aggregate` namespace.
Total word error rate, for instance, requires independently summing the numerators and denominators of the sample word error rates, instead of directly averaging them.
These metrics typically take a list or array of results as their single argument.

The `apricot`, `carla`, and `dapricot` metrics are effectively the `object_detection` metrics with parameters adapted to those respective scenarios.

| Name | Namespace | Description |
|-------|-------|-------|
| `categorical_accuracy` | `task.batch.categorical_accuracy` | Categorical Accuracy |
| `top_5_categorical_accuracy` | `task.batch.top_5_categorical_accuracy` | Top-5 Categorical Accuracy |
| `per_class_accuracy` | `task.batch.per_class_accuracy` | Categorical accuracy per class, as a list per class|
| `per_class_mean_accuracy` | `task.batch.per_class_mean_accuracy` | Mean categorical accuray per class |
| `word_error_rate` | `task.batch.word_error_rate` | Word error rate |
| `total_wer` | `task.batch.aggregate.total_wer` | Total word error rate |
| `entailment` | `task.batch.entailment` | Entailment language metric (contradiction, neural, entailment) |
| `total_entailment` | `task.aggregate.total_entailment` | Total entailment |
| `tpr_fpr` | `task.population.tpr_fpr` | Return a dictionary containing TP, FP, TN, FN, TPR, FPR, TNR, FNR, and F1 Score (assuming binary inputs) |
| `video_tracking_mean_iou` | `task.batch.video_tracking_mean_iou` | Mean IOU between ground-truth and predicted boxes, averaged over all frames for a video |
| `video_tracking_mean_success_rate` | `task.batch.video_tracking_mean_success_rate` | Mean success rate averaged over all multiple IOU thresholds and all frames |
| `object_detection_AP_per_class` | `task.population.object_detection_AP_per_class` | Object Detection average precision per class |
| `object_detection_disappearance_rate` | `task.population.object_detection_disappearance_rate` | Object Detection Disappearance Rate |
| `object_detection_hallucinations_per_image` | `task.population.object_detection_hallucinations_per_image` | Object Detection Hallucinations Per Image |
| `object_detection_mAP` | `task.population.object_detection_mAP` | Object Detection mean average precision |
| `object_detection_misclassification_rate` | `task.population.object_detection_misclassification_rate` | Object Detection Misclassification Rate |
| `object_detection_true_positive_rate` | `task.population.object_detection_true_positive_rate` | Object Detection True Positive Rate | 
| `apricot_patch_targeted_AP_per_class` | `task.population.apricot_patch_targeted_AP_per_class` | OD metric applied to apricot scenario |
| `carla_od_AP_per_class` | `task.population.carla_od_AP_per_class` | OD metric applied to carla scenario |
| `carla_od_disappearance_rate`  | `task.population.carla_od_disappearance_rate` | OD metric applied to carla scenario |
| `carla_od_hallucinations_per_image` | `task.population.carla_od_hallucinations_per_image` | OD metric applied to carla scenario |
| `carla_od_misclassification_rate` | `task.population.carla_od_misclassification_rate` | OD metric applied to carla scenario |
| `carla_od_true_positive_rate` | `task.population.carla_od_true_positive_rate` | OD metric applied to carla scenario |
| `dapricot_patch_target_success` | `task.population.dapricot_patch_target_success` | OD metric applied to dapricot scenario |
| `dapricot_patch_targeted_AP_per_class` | `task.population.dapricot_patch_targeted_AP_per_class` | OD metric applied to dapricot scenario |
| `abstains` | `task.batch.abstains` | Takes a batch matrix of inputs and returns 1 for each row that are all 0 (abstention) |
| `identity_unzip` | `task.batch.identity_unzip` | Utility function for mapping from batches to list of samples |
| `identity_zip` | `task.aggregate.identity_zip` | Utility function for mapping from list of samples to single batch |


<br>

### Statistical Metrics

The statistical module provide metrics for measurement of statistical and information theoretic quantities.
It also contains helper functions to set up data structures (e.g., contingency tables) for computation with these metrics.

| Name | Namespace | Description |
|-------|-------|-------|
| `chi2_p_value` | `metrics.statistical.registered.chi2_p_value` | Chi Squared Value  |
| `fisher_p_value` | `metrics.statistical.registered.fisher_p_value` | Fisher P-Value |
| `spd` | `metrics.statistical.registered.spd` | Statistical Parity Difference |
| `filter_perplexity_fps_benign` | `metrics.statistical.registered.filter_perplexity_fps_benign` | Perplexity of Filtered Distribution |
| `perplexity` | `metrics.statistical.registered.perplexity` | P-to-Q Perplexity |
| `kl_div` | `metrics.statistical.registered.kl_div` | KL Divergence |
| `cross_entropy` | `metrics.statistical.registered.cross_entropy` | Distributional Cross Entropy |
| `class_bias` | `metrics.statistical.registered.class_bias` | Class Bias |
| `majority_mask` | `metrics.statistical.registered.majority_mask` | Binary mask indicating whether a sample is in the majority of the distribution |
| `class_majority_mask` | `metrics.statistical.registered.class_majority_mask` | Majority mask with majority membership considered on a per-class basis |

<br>

### Poisoning Metrics

The poisoning scenarios can be configured to measure fairness across classes.
This code is in `armory.metrics.poisoning`, but doesn't constitute typical metric definitions.
Instead, it uses metrics from `statistical` (`chi2_p_value` and `spd`) to measure fairness.
In particular, it uses clustering from the activations of an explanatory model to determine samples that are in the "majority" or "minority" of the distribution, via `class_majority_mask` in `statistical`.
The statistical metrics are then used to compare the fairness across these subpopulations.

This module mostly contains code to load explanatory models, generate activations, and route the correct data as inputs to the statistical metrics.
For more information, see [poisoning](poisoning.md).
<br>

### Custom Metrics

In order to include custom metrics in configs, there are a few different steps.
NOTE: only perturbation and task metrics are loadable directly from the config at this time.

First, you will need to ensure that your custom metric code can be imported in the armory execution environment; see [external repos](external_repos.md) for more details on that.

In order for your metric to get loaded, it must be retrievable via the following function:
```
metrics.get(name)
```
where `name` is the `str` name of your function. Suppose your metric is defined in `my_project/metrics.py` as `hot_dog_ness`.

Using the custom metric requires providing the full `.`-separated path to the metric function in the config, e.g., `"my_project.metrics.hot_dog_ness"`.
In this case, `metrics.get("my_project.metrics.hot_dog_ness")` will try to import `hot_dog_ness` from `my_project.metrics`. Note the following caveats:
- This case will only work as intended if `hot_dog_ness` is a batchwise function that outputs a list (or array) of results, one per element in the batch
- By default, armory will try to calculate a mean from the output of the custom metric
- Should the name of the custom metric collide with any existing functions supported by armory, armory will throw an error notifying the user of the collision as well as request a name change for the custom metric

*Optional* An alternative is to use one of the existing decorators in `task` or `perturbation` to register your metric. This is useful for applying a custom metric as a non-batchwise operation and suppressing the mean calculation for outputs with specific formats.
These decorators, their associated namespaces, and the intended APIs of the metric functions they decorate, are:
- `metrics.perturbation.elementwise` - `metrics.perturbation.element` - takes a single pair of `x_i` and `x_adv_i` and returns a single perturbation distance for that element.
- `metrics.perturbation.batchwise` - `metrics.perturbation.batch` - takes a batch of `x` and `x_adv` and returns a list of results, one per data element.
- `metrics.task.elementwise` - `metrics.task.element` - takes a single pair of `y_i` and `y_pred_i` and returns a single result for that element.
- `metrics.task.batchwise` - `metrics.task.batch` - takes a batch of `y` and `y_pred` and returns a list of results, one per data element.
- `metrics.task.populationwise` - `metrics.task.population` - takes lists of `y` and `y_pred` across the entire dataset and computes a single set of metrics, such as mAP.
- `metrics.task.aggregator` - `metrics.task.aggregate` - takes a list of results from a batchwise metric and performs a non-trivial aggregation, such as for calculating total word error rate.

For instance, if you were adding an accuracy metric for task results, you could do:
```
from armory import metrics
@metrics.task.elementwise
def my_accuracy_metric(y_i, y_pred_i):
    return y_i == np.argmax(y_pred_i)
``` 

Armory performs all built-in metric operations as batches, not as individual elements, so using the `elementwise` decorators will also produce a batchwise version of it that loops through the individual elements and provides a batchwise result.
NOTE: when armory uses `get`, it will get the batchwise version of a metric.

Once annotated, these will also be `.`-addressable using their respective namespaces.
In the above example, you can get at `my_accuracy_metric` via the `metrics.task.element` namespace:
```python
metrics.task.element.my_accuracy_metric
assert metrics.task.element.my_accuracy_metric is my_accuracy_metric
```
You can also get the batchwise version of it via:
```python
batchwise_my_accuracy_metric = metrics.task.batch.my_accuracy_metric
```

All non-elementwise metrics are registered in the `supported` namespace, which can be looked at via `metrics.supported` or `metrics.common.supported`.

If the function is decorated with a `populationwise` decorator, then all of the results will be stored in a list until the end of the scenario, then passed to the metric for processing.
We do not support populationwise for perturbation metrics as this would require storing a potentially very large set of input and perturbed input data.

Another useful decorator currently only supported in `task` is the `aggregator`, which can take intermediate results from a batchwise metric and aggregate them together non-trivially.

For instance, if have a metric (like word error rate) that returns two values that form a fraction, and you want the aggregate measure to not be the mean over fractions, but the fraction of the sums of the numerator and denominator, that could look like the following:
```python
@metrics.task.elementwise
def fraction(y_i, y_pred_i) -> (int, int):
    ...
    return numerator / denominator

@metrics.task.aggregator
# NOTE: the input will be a list (or numpy array) of each element-wise result from the underlying function
def fraction_aggregator(list_of_fraction_tuples) -> (int, int):
    total_numerator, total_denominator = 0, 0
    for numerator, denominator in list_of_fraction_tuples:
        total_numerator += numerator
        total_denominator += denominator
    return total_numerator, total_denominator
```

In order for the aggregator to be automatically used in armory scenarios, it needs to be registered:
```
metrics.task.map_to_aggregator("fraction", "fraction_aggregator")
```
The `map_to_aggregator` maps the `str` name of the batchwise function (`fraction` is here implictly via the elementwise decorator making a batchwise version) to the `str` name of the aggregator function.

To test, you should be able to do:
```
aggregator_name = metrics.task.get_aggregator_name("fraction")
assert aggregator_name is not None
aggregator = metrics.get(aggregator_name)
assert aggregator is fraction_aggregator
```
This sequence of operations is essentially what armory does to resolve the aggregator.

If an aggregator is not linked to a metric, then it will default to either no aggregator (if `means` is `false` in the config) or to `np.mean` (if `means` is `true` in the config).
If `np.mean` fails, no aggregation result will be recorded, but a warning will be logged.

#### Log-Based Reporting

In addition to being output to results json files, results are also logged to the screen at the `"METRIC"` log level, which is between `"PROGRESS"` and `"INFO"`; see [logging](logging.md).
By default, metric results are formatted as follows: `f"{np.mean(result):.3}"`.
If this results in an error (e.g., due to result not being a number or array of numbers), then it defaults to `f"{result}"`.

In order to provide custom formatting for results logged to screen, you will need to implement and register a formatter for your function.
Following the `fraction` example above, you can do the following:
```python
@metrics.result_formatter("fraction")
def fraction_formatter(result) -> str:
    numerator, denominator = result
    return f"{numerator} / {denominator}"
```
The arg provided to the `result_formatter` decorator is the name of the metric you would like to associate with this formatter, which should return a `str`.
If you would like to associate additional metrics with this formatter (e.g., to use it for the aggregation function as well), you can call it directly as follows:
```python
metrics.result_formatter("fraction_aggregator")(fraction_formatter)
```

#### Class-based Metrics

The previous description assumes that the metric is a callable function.
It is sometimes necessary or helpful for a metric to be contained in a class.
When loading, if the target returned by `metrics.get` is a class, it will attempt to instantiate the class (without args or kwargs), and use the instantiated callable object as the metric function.
Otherwise, it should operate just like a simple function metric.

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
Meters and writers can be instantiated in your initial setup (please refer to [User Initialization](./scenarios.md#user-initialization) for more details about using the `user_init` block), and can be connected before probes are constructed.

#### Direct Recording

To capture one-off values or values that do not require metric measurement, you can push a record to the hub directly using its `record` method:
```
hub = get_hub()
name = "my_record"
result = 17
hub.record(name, result)
```
This will push a record to all default writers (including the `ResultsWriter` in standard scenarios) with that information.
To send it to an additional writer or writers, you can supply them with the `writers` kwargs, which can take a single writer or an iterable of writers.
To not send it to the default writers, set the `use_default_writers` kwarg to `False`.
For instance:
```
my_writer = PrintWriter()
hub.record(name, result, writers=my_writer, use_default_writers=False)
```
If `writers` is empty or None and `use_default_writers` is False, no record will be sent and a warning will be logged.

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

A more succinct way of doing this, which also handles the case when only batches of `w` and `z` are supplied to the meter, is to use `GlobalMeter`:
```python
meter = GlobalMeter(
    "mean_meter",  # actual recorded name
    np.mean,
    "my_probe.w",
    "my_probe.z",
    final_kwargs=None,
)
The `GlobalMeter` assumes that inputs are batches, so if multiple batches are passed, they will be concatenated along the first axis.
Note that if multiple batches are passed, all variables from a specific batch must be passed to the meter before moving to the next batch, in order to avoid dropping.

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
