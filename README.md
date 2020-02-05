<div align="center">
  <img src="tools/static_content/logo.png" width="50%" title="ARMORY logo">
</div>

-----------------
[![Travis Nightly](https://travis-ci.com/twosixlabs/armory.svg?token=mDXSPweWiXNcpsV8rz4z&branch=master)](https://travis-ci.com/twosixlabs/armory)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

ARMORY is a test bed for running scalable evaluations of adversarial defenses. 
Configuration files are used to launch local or cloud instances of the ARMORY docker 
container. Models, datasets, and evaluation scripts can be pulled from external 
repositories or from the baselines within this project.

# Setup
Python 3.6+ is required.

### Installation
```
pip install git+https://github.com/twosixlabs/armory.git
```

### Docker
Armory is intended to be a lightweight python package which standardizes all evaluations
inside a docker container. Users are encouraged to use the available image on dockerhub 
,however if there are issues downloading (e.g. proxy) it can be built within this repo:
```
docker build -t twosixarmory/armory:0.1.1 .
```


# Usage
Since ARMORY launches Docker containers, the python package must be ran on system host.

Typically evaluations are ran using the [run_evaluation script](run_evaluation.py). 
Please [see example configuration files](examples/).


Debugging evaluations can be performed interactively by passing `--interactive` and 
following the instructions to attach to the container in order to use pdb or other
interactive tools.