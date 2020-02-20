<div align="center">
  <img src="https://github.com/twosixlabs/armory/blob/master/tools/static_content/logo.png" width="50%" title="ARMORY logo">
</div>

-----------------
[![GitHub CI](https://github.com/twosixlabs/armory/workflows/GitHub%20CI/badge.svg)](https://github.com/twosixlabs/armory/actions?query=workflow%3A%22GitHub+CI%22)
[![PyPI Status Badge](https://badge.fury.io/py/armory-testbed.svg)](https://pypi.org/project/armory-testbed)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/armory-testbed)](https://pypi.org/project/armory-testbed)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

ARMORY is a test bed for running scalable evaluations of adversarial defenses. 
Configuration files are used to launch local or cloud instances of the ARMORY docker 
container. Models, datasets, and evaluation scripts can be pulled from external 
repositories or from the baselines within this project.

# Installation
``` 
pip install armory-testbed
```

Upon installing armory, a directory will be created at `~/.armory`. This user 
specific folder is the default directory for downloaded datasets and evaluation 
outputs. Defaults can be changed by editing `~/.armory/config.json`

# Usage

ARMORY works by running an evaluation configuration file within the armory docker 
ecosystem. To do this, simply run `armory run <path_to_evaluation.json>`. 
Please [see example configuration files](examples/) for runnable configs.

The current working directory and armory installation directory will be mounted 
inside the container and the `armory.eval.Evaluator` class will proceed to run the 
evaluation script that is written in the `evaluation['eval_file']` field of the 
config.

For more detailed information on the evaluation config file please see the 
[documentation](examples/README.md).

Note: Since ARMORY launches Docker containers, the python package must be ran on system host.

As an example:
```
pip install armory-testbed
git clone https://github.com/twosixlabs/armory-example.git
cd armory-example
armory run example_config.json
```

### Interactive Debugging of Evaluations
Debugging evaluations can be performed interactively by passing `--interactive` and 
following the instructions to attach to the container in order to use pdb or other
interactive tools. There is also support for `--jupyter` which will open a port on 
the container and allow notebooks to be ran inside the armory environment.

### Custom Attacks and Defenses
Our evaluations are created so that attacks and defenses may be 
interchanged. To do this we standardize all attacks and defenses as subclasses of 
their respective implementations in [adversarial-robustness-toolbox](https://github.com/IBM/adversarial-robustness-toolbox)

# Docker
Armory is intended to be a lightweight python package which standardizes all evaluations
inside a docker container. Docker images will be pulled as needed when evaluations are 
ran.

However if there are issues downloading the images (e.g. proxy) they can be built 
within this repo:
```
version=$(python -c "import armory; print(armory.__version__)")
docker build --build-arg armory_version=${version} --target armory-tf1 -t twosixarmory/tf1:${version} .
docker build --build-arg armory_version=${version} --target armory-tf2 -t twosixarmory/tf2:${version} .
docker build --build-arg armory_version=${version} --target armory-pytorch -t twosixarmory/pytorch:${version} .
```

### Docker Mounts
By default when launching an ARMORY instance the current working directory will be mounted
as your default directory.This enables users to run modules from ARMORY baselines, 
as well as modules from the user project.

### Docker Setup
Depending on the task, docker memory for an ARMORY container must be at least 8 GB to run properly (preferably 16+ GB).
On Mac and Windows Desktop versions, this defaults to 2 GB. See the docs to change this:
* [Mac](https://docs.docker.com/docker-for-mac/)
* [Windows](https://docs.docker.com/docker-for-windows/)

### Docker Cleanup
Running `armory download-all-data` will download new Docker images, but will not clean up old images.

To download new images and clean up old images:
```
armory clean
```
If containers are currently running that use the old images, this will fail.
In that case, either stop them with first or run:
```
armory clean --force
```

To display the set of current images:
```
docker images
```
To manually delete images, see the docs for [docker rmi](https://docs.docker.com/engine/reference/commandline/rmi/).

In order to see the set of containers that are running:
```
docker ps
```
ARMORY will attempt to gracefully shut down all containers it launches;
however, certain errors may prevent shutdown and leave running containers.
To shut down these containers, please see the docs for [docker stop](https://docs.docker.com/engine/reference/commandline/stop/) and [docker kill](https://docs.docker.com/engine/reference/commandline/kill/).
