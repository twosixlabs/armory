# Getting Started

## Installation
Armory can be installed from PyPi:
```
pip install armory-testbed
```

When a user runs a given configuration file, the necessary docker image, datasets and 
model weights will be pulled as needed. We do have convenience functions to download 
all images, datasets and model weights for a scenario set release. This can take a 
while so you may want to run it overnight:
```
git clone https://github.com/twosixlabs/armory-example.git
cd armory-example
armory download official_scenario_configs/scenarios-set1.json
```  

## Baseline models
The armory package contains several framework specific baseline models that can be used
during evaluation. Please see our documentation on baseline models for more information 
about what is available and what pretrained weights can be pulled from S3:

[Baseline Model Docs](baseline_models.md)

## Running an evaluation
Evaluations are typically ran though the use of configuration files. See the 
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
current working directory project. This is supported through the `external_github_repo`
field in the configuration file. At launch, the repository will be pulled into the 
container and placed on the PYTHONPATH so it can be utilized.

This functionality supports public and private GITHUB repositories. If you would like 
to pull in a private repository, you'll need to set a user token as an environment 
variable before running `armory run`.

```
export GITHUB_TOKEN="5555e8b..."
armory run <path/to/config.json>
```

Tokens can be created here: [https://github.com/settings/tokens](https://github.com/settings/tokens)
