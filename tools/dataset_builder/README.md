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
needs, you can incorporate your specific dataset using the `lcs` option discussed in [Usage / Native Mode](#Native-mode) below.

# Install, Configure, and Usage
This package does not necessarily need to be installed to be used, however it does have some python dependencies
that are necessary for execution.  To set this up:
 - Establish a python environment to use (e.g. create a new virtualenv, docker container, etc.)
 - inside the environment install the requirements using: `pip install -r requirements.txt`
 - run the `build.py` script to build the dataset(s)
 - if needed, run the `upload.py` script to upload the artifacts to armory s3

## Native Mode
Once you have established your local python environment (e.g. virtualenv), activate it and then run:
```bash
pip install -r requirements.txt
python build.py -h
```
to get a full set of instructions for building datasets.  Of particular use is building a 
`supported` dataset (e.g. `mnist`, `digit`, etc.) by using:
```bash
python build.py -ds mnist digit
```
or to build all `supported` datasets
```bash
python build.py -ds all
```
Note: you can/should use the `--clean` flag if you have built datasets previously.  This 
flag will remove old artifacts and redownload as necessary to give a "fresh" build.

If you have a local dataset class file or tfds dataset style directory at `[local_path]`, 
you can build a dataset from that directory using:
```bash
python build.py -lcs [local_path]
```

## Docker mode
Some usages might require containerization to isolate the build process, for which we provide
minimal support.  The [Dockerfile](./Dockerfile) included here can be utilized by, first, 
building the image:
```bash
docker build --force-rm -t dsbuild .
```
and then using that image:
```bash
docker run --rm dsbuild python build.py -ds mnist
```
**NOTE**: the command above will construct and test the dataset but then, after the container exits, 
all the artifacts will be lost.  A better way is to do this interactively:
```bash
docker run -it --rm dsbuild bash
# This will put you in a bash shell inside the container
python build.py -ds mnist
# ... Do what you would like with the results
```
Alternatively, you can mount a drive to the container and then use the `-o, --output-directory` 
options to point to that directory to store the artifacts. 

## Uploading Files to Armory s3
Once datasets are built, you can use the `upload.py` script to upload the `.tar.gz` file 
containing the dataset artifacts to the armory s3 datastore.  For more information see:
```bash
python upload.py -h
```

# NOTES:
 - TFDS Build CLI requires that the python file defining the builder class be in a directory 
   that also contains an `__init__.py`.  Furthermore, TFDS will use the class name and
   `VERSION` attribute of the builder class to name and construct the resulting dataset folder 
   that contains the `tfrecord` files.  Each class file has to be in its own directory with a 
   `__init__.py` file or else the `tfds build ...` will fail.
 - TFDS Build CLI also requires exactly one `tfds.core.GeneratorBasedBuilder` class per file. 
   It uses the name of that class as the dataset name (aka folder name) and separates the 
   CamelCase words with underscores. Note:  It is important to pay attention to naming as it 
   is easy to create namespace clashes if the naming protocols are not adhered to.
 - Generally, TFDS uses the class name of the `tfds.core.GeneratorBasedBuilder` class in the 
   class file to name the dataset, which allows for the name of the class file to be different,
   however, this `build.py` script enforces the naming of the class file and the class itself
   be consistent.  
   
