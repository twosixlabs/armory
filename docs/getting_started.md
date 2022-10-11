# Getting Started

## Installation
Armory can be installed from PyPi:
```
pip install armory-testbed[framework-flavor]
```

Where `framework-flavor` is one of `tensorflow`, `pytorch` or `deepspeech`
as described below in [the armory flavors](#the-armory-flavors).

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
[config file documentation](/docs/configuration_files.md) for information regarding the
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
information can be found in the [external repo documentation](/docs/external_repos.md).

## the armory flavors

Armory supports multiple frameworks:

  - tensorflow
  - pytorch
  - deepspeech

In releases prior to 0.16, there was a complex set of `*-requirements.txt` files
that were needed to provision the python environment for the various frameworks.
As of Armory 0.16, these have all been consolidated into the standard
`pyproject.toml` at the repository root.

We now use the optional-dependencies feature of pyproject which requires
the selection of a flavor to be specified at install time.  For example:

     pip install armory-testbed

installs no framework libraries so will fail to run any framework dependent code. Future
armory releases may use this flavorless base. To install the tensorflow flavor:

     pip install armory-testbed[tensorflow]

which installs the libraries needed for tensorflow evaluations. Similarly,

    pip install armory-testbed[pytorch]

or

    pip install armory-testbed[deepspeech]

depending on the framework you want to use. We don't recommend trying to
install multiple frameworks at the same time as this may lead to dependency
conflicts. So

    pip install armory-testbed[tensorflow,pytorch]

is unsupported and may not even install.

## additional flavors

You can freely add `jupyterlab` to the flavor list to as needed, for example

    pip install armory-testbed[tensorflow,jupyterlab]

People developing armory will likely want to add the `developer` flavor to their
set:

    pip install armory-testbed[deepspeech,developer,jupyterlab]

Developers who are creating new Armory datasets will need

    pip install armory-testbed[datasets-builder]

## editable installs

As before, the `--editable` flag can be used to install in editable mode
which is often useful for development.

The `.` installation target is also supported, but even that requires
a flavor specification. That is, where you might have previously run

    pip install --editable .

you now need to specify a flavor:

    pip install --editable .[pytorch]
