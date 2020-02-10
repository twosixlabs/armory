<div align="center">
  <img src="https://github.com/twosixlabs/armory/blob/master/tools/static_content/logo.png" width="50%" title="ARMORY logo">
</div>

-----------------
[![Travis Nightly](https://travis-ci.com/twosixlabs/armory.svg?token=mDXSPweWiXNcpsV8rz4z&branch=master)](https://travis-ci.com/twosixlabs/armory)
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
At the moment our evaluations are created so that attacks and defenses may be 
interchanged. To do this we standardize all attacks and defenses as subclasses of 
their respective implementations in [adversarial-robustness-toolbox](https://github.com/IBM/adversarial-robustness-toolbox)

# Docker
Armory is intended to be a lightweight python package which standardizes all evaluations
inside a docker container. Users are encouraged to use the available images on 
dockerhub:
```
docker pull twosixarmory/tf1:0.3.3
docker pull twosixarmory/tf2:0.3.3
docker pull twosixarmory/pytorch:0.3.3
```

However if there are issues downloading the images (e.g. proxy) they can be built 
within this repo:
```
docker build --target armory-tf1 -t twosixarmory/tf1:0.3.3 .
docker build --target armory-tf2 -t twosixarmory/tf2:0.3.3 .
docker build --target armory-pytorch -t twosixarmory/pytorch:0.3.3 .
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
