# Configuration Files

All configuration files are verified against the jsonschema definition at run time:
[armory/utils/config_schema.json](https://github.com/twosixlabs/armory/blob/master/armory/utils/config_schema.json)

## Schema
```
`_description`: [String] Any description that describes the scenario evaluation
`adhoc`: [Object or null]
  {    
    Custom parameters that you can access within a scenario
  }
`attack`: [Object or null]
  {
    knowledge: [String] `white` or `black` knowledge    
    kwargs: [Object] Keyword arguments to pass to attack instatiation    
    module: [String] Python module to load attack from 
    name: [String] Name of the attack class to be instatiated
    use_label: [Bool] Default: False. Whether attack should use the true label when 
          attacking the model. Without this, it is not possible to drive the accuracy 
          down to 0% when the model has misclassifications.
    type: [Optional String]: in <`preloaded`|`patch`|`sweep`>.
  }
`dataset`: [Object]
  {
    batch_size [Int]: Number of samples to include in each batch
    module: [String] Python module to load dataset from 
    name: [String] Name of the dataset function
    framework: [String] Framework to return Tensors in. <`tf`|`pytorch`|`numpy`>. `numpy` by default.
    train_split: [Optional String] Training split in dataset. Typically defaults to `train`. Can use fancy slicing via [TFDS slicing API](https://www.tensorflow.org/datasets/splits#slicing_api)
    eval_split: [Optional String] Eval split in dataset. Typically defaults to `test`. Can use fancy slicing via [TFDS slicing API](https://www.tensorflow.org/datasets/splits#slicing_api)
    class_ids: [Optional Int or List[Int]] Class ID's to filter the dataset to. Can use a numeric list like [1, 5, 7] or a single integer.
    index: [Optional String or Object] Index into the post-sorted (and post-filtered if class_ids is enabled) eval dataset. Can use a numeric list like [1, 5, 7] or a simple slice as a string, like "[3:6]" or ":100".
  }
`defense`: [Object or null]
  {
    kwargs: [Object] Keyword arguments to pass to defense instatiation    
    module: [String] Python module to load defense from 
    name: [String] Name of the defense class to be utilized
    type: [String] Type of defense which flags how it should be used. One of <Preprocessor, Postprocessor, Trainer, Transformer, PoisonFilteringDefence>
  }
`metric`: [Object or null]
  {
    means: [Bool] Boolean to caculate means for each task in logging / output
    perturbation: [String] Perturbation metric to calculate for adversarial examples
    record_metric_per_sample: [Bool] Boolean to record metric for every sample in save in output
    task: [List[String]] List of task metrics to record (e.g. categorical_accuracy)
    profiler_type: [Optional String or null] Type of computational resource profiling desired for scenario profiling. One of <basic, deterministic> or null
  }
`model`: [Object]
  {
    fit: [Bool] Boolean to train the model or not
    fit_kwargs: [Object] Keyword arguments to pass to `fit_generator` or `fit`
    module: [String] Python module to load model from 
    name: [String] Name of the function to return ART classifier
    model_kwargs: [Object] Keyword arguments to load model function
    weights_file: [String or null] Name of pretrained weights file. Will be downloaded from S3 if available
    wrapper_kwargs: [Object] Keyword arguments to ART wrapper function
  }
`scenario`: [Object]
  {
    kwargs: [Object] Keyword arguments to pass to Scenario instatiation
    module: [String] Python module to load scenario from 
    name: [String] Name of the scenario class to be ran
    export_batches: [Optional Int or Bool] Number of batches of data to export
  }
`sysconfig` [Object]
  {
    docker_image: [String or null] Docker image name and tag to run scenario in
    external_github_repo: [String or null or Object] External github repository(s) to download and place on PYTHONPATH within container
    external_github_repo_pythonpath: [String or null or Object] Relative path(s) in the repo directory to add to PYTHONPATH within container
    gpus: [String]: Which GPUs should the docker container have access to. "all" or comma sperated list (e.g. "1,3")
    local_repo_path: [String or null or Object] Local github repository path(s) to place on PYTHONPATH within container
    output_dir: [Optional String]:  Add an optional output directory prefix to the default output directory name.
    output_filename: [Optional String]: Optionally change the output filename prefix (from default of scenario name)  
    use_gpu: [Boolean]: Boolean to run container as nvidia-docker with GPU access
  }
`user_init`: [Object or null]
  {
    module: [String] Python module to import before scenario loading but after scenario initialization
    name: [String or null] Name of the function to call after module import (optional)
    kwargs: [Object or null] Keyword arguments to provide for function call (optional)
  }
```


### Example Configuration File:
```
{
    "_description": "Baseline cifar10 image classification",
    "adhoc": null,
    "attack": {
        "knowledge": "white",
        "kwargs": {
            "batch_size": 1,
            "eps": 0.031,
            "eps_step": 0.007,
            "max_iter": 20,
            "num_random_init": 1,
            "random_eps": false,
            "targeted": false,
            "verbose": false
        },
        "module": "art.attacks.evasion",
        "name": "ProjectedGradientDescent",
        "use_label": true
    },
    "dataset": {
        "batch_size": 64,
        "framework": "numpy",
        "module": "armory.data.datasets",
        "name": "cifar10"
    },
    "defense": null,
    "metric": {
        "means": true,
        "perturbation": "linf",
        "record_metric_per_sample": false,
        "task": [
            "categorical_accuracy"
        ]
    },
    "model": {
        "fit": true,
        "fit_kwargs": {
            "nb_epochs": 20
        },
        "model_kwargs": {},
        "module": "armory.baseline_models.pytorch.cifar",
        "name": "get_art_model",
        "weights_file": null,
        "wrapper_kwargs": {}
    },
    "scenario": {
        "kwargs": {},
        "module": "armory.scenarios.image_classification",
        "name": "ImageClassificationTask"
    },
    "sysconfig": {
        "docker_image": "twosixarmory/armory",
        "external_github_repo": null,
        "gpus": "all",
        "output_dir": null,
        "output_filename": null,
        "use_gpu": false
    }
}
```

### attack config "type" field
The supported values for the `"type"` field in attack configs are as follows: 
<`preloaded`|`patch`|`sweep`>. If none of the cases below apply, the field 
does not need to be included.

1. `"preloaded"`: This value is specified when using an adversarial dataset (e.g. APRICOT) where no 
perturbations should be applied to the inputs.

2. `"patch"`: Some ART attacks such as `"AdversarialPatch"`  or `"RobustDPatch"` have a `generate()` method
which returns a patch rather than the input with patch. When using such an attack in an Armory scenario,
setting `attack_config["type"]` to `"patch"` will enable an Armory wrapper class with an updated 
`generate()` which applies the patch to the input.

2. `"sweep"`: To enable "sweep" attacks, see the instructions in [sweep_attacks.md](sweep_attacks.md).

### Use with Custom Docker Image

To run with a custom Docker image, replace the `["sys_config"]["docker_image"]` field
to your custom docker image name `<your_image/name:your_tag>`.

### Specifying kwargs for metric functions
Some metric functions in [armory/utils/metrics.py](../armory/utils/metrics.py) receive kwargs, e.g.
`iou_threshold` in the case of `object_detection_AP_per_class()`. To modify the kwarg, specify 
`"task_kwargs"` in the `"metric"` portion of the config file as such:

```json
"metric":  {
        "task": [
            "object_detection_AP_per_class",
            "object_detection_true_positive_rate"
        ],
        "task_kwargs": [{"iou_threshold": 0.3}, {}]
}
```
Note that the length of `"task_kwargs"` should be equal to that of `"task"`, as `task_kwargs[i]` corresponds
to `task[i]`.

### Exporting Data
Please see [exporting_data.md](exporting_data.md).

### Additional configuration settings for poisoning scenario

Some settings specific to the poisoning scenario are not applicable to the other 
scenarios and are thus found in "adhoc" subfield of the configuration file.

For a poison filtering defense, Armory supports using a model for filtering that 
differs from the model used at training time. The model used at training time should 
still be stored in the field "model" as described in the config schema. However, if a 
different model is used for the filtering defense, it should be entered in the "ad-hoc" 
field of the configuration file under the subfield "defense_model," with the number of
epochs of training under the subfield "defense_model_train_epochs." A concrete example
of a configuration with this field is available in the armory-example
[repo](https://github.com/twosixlabs/armory-example/tree/master/example_scenario_configs).

### sysconfig and command line arguments

Parameters specified in the "sysconfig" block will be treated as if they were passed
as arguments to `armory` for example a configuration block like
```json
{
  "sysconfig": {
    "num_eval_batches": 5,
    "skip_benign": true
  }
}
```
will cause armory to act as if you had run it as
```
armory run scenario.json --num-eval-batches 5 --skip-benign
```
However, arguments actually specified on the command line will take precedence,
so if you execute, using the same configuration file
```
armory run scenario.json --num-eval-batches 100
```
Then the command line will override the sysconfig and 100 batches (not 5) will
be run. In this example, `--skip-benign` will also be true because it is
in the sysconfig block.

No matter whether these attributes are specified on the command line, in sysconfig,
or both, the output file will record the attributes as executed, so you have a
record of how the evaluation ultimately ran.

The [full specification of command line arguments][cmdline] is available.

  [cmdline]: command_line.md
