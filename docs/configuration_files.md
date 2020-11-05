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
  }
`dataset`: [Object]
  {
    batch_size [Int]: Number of samples to include in each batch
    module: [String] Python module to load dataset from 
    name: [String] Name of the dataset function
    framework: [String] Framework to return Tensors in. <`tf`|`pytorch`|`numpy`>. `numpy` by default.
    train_split: [Optional String] Training split in dataset. Typically defaults to `train`. Can use fancy slicing via [TFDS slicing API](https://www.tensorflow.org/datasets/splits#slicing_api)
    eval_split: [Optional String] Eval split in dataset. Typically defaults to `test`. Can use fancy slicing via [TFDS slicing API](https://www.tensorflow.org/datasets/splits#slicing_api)
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
    profiler_type: [Optional String] Type of computational resource profiling desired for scenario profiling. One of <Basic, Deterministic>
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
```


### Example Configuration File:
```
{
    "_description": "Example configuration",
    "adhoc": null,
    "attack": {
        "knowledge": "white",
        "kwargs": {
            "eps": 0.2
        },
        "module": "art.attacks",
        "name": "FastGradientMethod",
        "use_label": false
    },
    "dataset": {
        "batch_size": 64,
        "module": "armory.data.datasets",
        "name": "cifar10",
        "framework": "numpy"
    },
    "defense": null,
    "metric": {
        "means": true,
        "perturbation": "linf",
        "record_metric_per_sample": true,
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
        "name": "ImageClassification"
    },
    "sysconfig": {
        "docker_image": "twosixarmory/pytorch:0.6.0",
        "external_github_repo": "twosixlabs/armory-example@master",
        "gpus": "all",
        "use_gpu": false
    }
}
```

### Use with Custom Docker Image

To run with a custom Docker image, replace the `["sys_config"]["docker_image"]` field
to your custom docker image name `<your_image/name:your_tag>`.

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
