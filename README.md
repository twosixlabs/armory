<div align="center">
  <img src="tools/static_content/logo.png" width="50%" title="ARMORY logo">
</div>

-----------------
[![Travis Nightly](https://travis-ci.com/twosixlabs/armory.svg?token=mDXSPweWiXNcpsV8rz4z&branch=master)](https://travis-ci.com/twosixlabs/armory)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

Adversarial Robustness Evaluation Test Bed

# Installation
Python 3.6+ is required.
```
pip install git+https://github.com/twosixlabs/armory.git
```

# Docker
Docker is required to run ARMORY.

Docker memory for an ARMORY container must be at least 8 GB to run properly (preferably 16+ GB).
On Mac and Windows Desktop versions, this defaults to 2 GB. See the docs to change this:
* [Mac](https://docs.docker.com/docker-for-mac/).
* [Windows](https://docs.docker.com/docker-for-windows/)
* [Linux](https://docs.docker.com/install/linux/docker-ce/ubuntu/)

Armory is intended to be a lightweight python package which standardizes all evaluations
inside a docker container. 
```
docker build -t twosixlabs/armory:0.1.1 .
```
Since ARMORY launches Docker containers, the python package must be ran on system host.

# Evaluation
Typically evaluations are ran using the [run_evaluation script](run_evaluation.py). 
Please [see example configuration files](examples/).

# Datasets
We have standardized datasets for Armory that subclass TensorFlow Datasets:
https://github.com/tensorflow/datasets

These datastructures support coversion to numpy arrays so they will work for all 
frameworks that we support.


# APIs
* Data
* Adversarial data
* Kubeflow AWS
