# ARMORY
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

Adversarial Robustness Evaluation Platform

# Installation
Python 3.6+ is required.
```
pip install git+https://github.com/twosixlabs/armory.git
```

# Docker
Armory is intended to be a lightweight python package which standardizes all evaluations
inside a docker container. 
```
docker build -t twosixlabs/armory:0.1 .
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
