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

# Formatting
All contributions to the repository must be formatted with [black](https://github.com/psf/black).
```
black .
```

All JSON files committed to the repository must be formatted with Python's [json.tool](https://docs.python.org/3/library/json.html#module-json.tool)
with the `--sort-keys` argument:
```
python -m json.tool input_file.json input_file.json --sort-keys
```

