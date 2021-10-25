# Getting Started

## Installation
Armory can be installed from PyPi:
```
pip install armory-testbed
```

When a user runs a given configuration file, the necessary docker image, datasets and 
model weights will be pulled as needed. We do have convenience functions to download 
all images, datasets and model weights for a scenario set release. This can take a 
while, so you may want to run it overnight:
```
git clone https://github.com/twosixlabs/armory-example.git
cd armory-example
# First set of examples:
armory download scenario_download_configs/scenarios-set1.json
# Second set of scenarios:
armory download scenario_download_configs/scenarios-set2.json
```  
If you are not using Docker, then add `--no-docker`: 
```
armory download scenario_download_configs/scenarios-set*.json --no-docker
```

If you want to download with a specific image, use:
```
armory download --docker-image <image tag> scenario_download_configs/scenarios-set*.json
```

## Baseline models
The armory package contains several framework specific baseline models that can be used
during evaluation. Please see our documentation on baseline models for more information 
about what is available and what pretrained weights can be pulled from S3:

[Baseline Model Docs](baseline_models.md)

## Running an evaluation
Evaluations are typically run through the use of configuration files. See the 
[config file documentation](configuration_files.md) for information regarding the 
schema and what the fields refer to.

To run a configuration:
```
git clone https://github.com/twosixlabs/armory-example.git
cd armory-example
armory run official_scenario_configs/cifar10_baseline.json
```

## External Repos
You may want to include code from an external repository that is outside of your 
current working directory project. This is fully supported by Armory and more 
information can be found in the [external repo documentation](external_repos.md).