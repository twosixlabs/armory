ARMORY DATASET Builder Package
==============================
<div align="center">
  <img src="https://github.com/twosixlabs/armory/blob/master/tools/static_content/logo.png" width="50%" title="ARMORY logo">
</div>

-----------------
[![GitHub CI](https://github.com/twosixlabs/armory/workflows/GitHub%20CI/badge.svg)](https://github.com/twosixlabs/armory/actions?query=workflow%3A%22GitHub+CI%22)
[![PyPI Status Badge](https://badge.fury.io/py/armory-testbed.svg)](https://pypi.org/project/armory-testbed)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/armory-testbed)](https://pypi.org/project/armory-testbed)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docs](https://readthedocs.org/projects/armory/badge/)](https://armory.readthedocs.io/en/latest/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

# Overview

The ARMORY dataset builder packages is a set of helper methods based around the tensorflow_datasets (TFDS) version 4 API/CLI.
The basic process of constructing datasets follows:
   1.  Identify Source Files for the dataset (e.g. Images, labels, etc.)
   2.  Construct a TFDS builder class (e.g. `tfds.core.GeneratorBasedBuilder`, etc.)
         * For more information on this process see: [TFDS Generator Based Builder Docs](https://www.tensorflow.org/datasets/api_docs/python/tfds/core/GeneratorBasedBuilder)
   3.  use `tfds build` CLI to construct the necessary artifacts (e.g. `tfrecord` files and meta-data files)

Armory provides support for a variety of datasets out of the box, which include datasets that are pre-packaged from TFDS as 
well as custom datasets that have been assembled by the armory team from various sources.  If these do not meet your dataset
needs, you can incorporate your specific dataset using the `lcs` option discussed in [Usage](#Usage) below.

# Install & Configure
This package does not necessarily need to be installed to be used, however it does have some python dependencies
that are necessary for execution.  To set this up:
 - Establish a python environment to use (e.g. create a new virtualenv)
 - inside the environment install the requirements using: `pip install -r requirements.txt`

# Usage
To run the builder use:
    python run.py -h

# NOTES:
 - TFDS Build CLI requires that the python file defining the builder class be in a directory that also contains
   an `__init__.py`.  Furthermore, TFDS will use the class name and `VERSION` attribute of the builder class to 
   name and construct the resulting dataset folder that contains the `tfrecord` files.  Each class file has to be in its own directory with a `__init__.py` file or 
   else the `tfds build ...` will fail
   
 - run with `python run.py -ds all --clean` for all supported datasets

TFDS build requires 1 `tfds.core.GeneratorBasedBuilder` class per file.  
it uses the name of that class as the dataset name (aka folder name)

It separates CamelCase class names with underscores

We will enforce that the class file name must match the class name