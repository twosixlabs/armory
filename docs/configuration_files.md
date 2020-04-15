# Configuration Files

All configuration files are verified against the jsonschema definition at run time:
[armory/utils/config_schema.json](https://github.com/twosixlabs/armory/blob/master/armory/utils/config_schema.json)

### Schema
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
  }
`dataset`: [Object]
  {
    batch_size [Int]: Number of samples to include in each batch
    module: [String] Python module to load dataset from 
    name: [String] Name of the dataset function
  }
`defense`: [Object or null]
  {
    kwargs: [Object] Keyword arguments to pass to defense instatiation    
    module: [String] Python module to load defense from 
    name: [String] Name of the defense class to be utilized
    type: [String] Type of defense which flags how it should be used. One of <Preprocessor, Postprocessor, Trainer, Transformer>
  }
`metric`: [Object or null]
  {
    means: [Bool] Boolean to caculate means for each task in logging / output
    perturbation: [String] Perturbation metric to calculate for adversarial examples
    record_metric_per_sample: [Bool] Boolean to record metric for every sample in save in output
    task: [List[String]] List of task metrics to record (e.g. categorical_accuracy)
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
    docker_image: [String] Docker image name and tag to run scenario in
    external_github_repo: [String or null] External github repository to download and place on PYTHONPATH within container
    gpus: [String]: Which GPUs should the docker container have access to. "all" or comma sperated list (e.g. "1,3")
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
        "name": "FastGradientMethod"
    },
    "dataset": {
        "batch_size": 64,
        "module": "armory.data.datasets",
        "name": "cifar10"
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