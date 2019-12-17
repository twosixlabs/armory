# ARMORY
Adversarial Robustness Evaluation Platform for DARPA GARD

# Installation
```
pip install git+https://github.com/twosixlabs/armory.git
```

# Docker
The docker container is required form running evaluations. 
```
docker build -t twosixlabs/armory:0.1 .
```

# Datasets
We have standardize datasets for Armory to subclass TensorFlow Datasets:
https://github.com/tensorflow/datasets

These datastructures support coversion to numpy arrays so they will work for all 
frameworks that we support.


# Evaluation
[See examples](examples/).

# APIs
* Data
* Adversarial data
* Kubeflow AWS

# Formatting
All contributions to the repository must be formatted with [black](https://github.com/psf/black).
```
black .
```
