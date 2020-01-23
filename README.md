# ARMORY
Adversarial Robustness Evaluation Platform for DARPA GARD

# Installation
Python 3.6+ is required.
```
pip install git+https://github.com/twosixlabs/armory.git
```

# Docker
The docker container is required for running evaluations.
```
docker build -t twosixlabs/armory:0.1 .
```
Since ARMORY launches Docker containers, the package must be ran on system host.

# Datasets
We have standardized datasets for Armory to subclass TensorFlow Datasets:
https://github.com/tensorflow/datasets

These datastructures support coversion to numpy arrays so they will work for all 
frameworks that we support.


# Evaluation
[See examples](examples/).

# APIs
* Data
* Adversarial data
* Kubeflow AWS
