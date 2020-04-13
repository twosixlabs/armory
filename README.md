<div align="center">
  <img src="https://github.com/twosixlabs/armory/blob/master/tools/static_content/logo.png" width="50%" title="ARMORY logo">
</div>

-----------------
[![GitHub CI](https://github.com/twosixlabs/armory/workflows/GitHub%20CI/badge.svg)](https://github.com/twosixlabs/armory/actions?query=workflow%3A%22GitHub+CI%22)
[![PyPI Status Badge](https://badge.fury.io/py/armory-testbed.svg)](https://pypi.org/project/armory-testbed)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/armory-testbed)](https://pypi.org/project/armory-testbed)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

# Overview

ARMORY is a test bed for running scalable evaluations of adversarial defenses. 
Configuration files are used to launch local or cloud instances of the ARMORY docker 
containers. Models, datasets, and evaluation scripts can be pulled from external 
repositories or from the baselines within this project. 

Our evaluations are created so that attacks and defenses may be 
interchanged. To do this we standardize all attacks and defenses as subclasses of 
their respective implementations in the [adversarial-robustness-toolbox](https://github.com/IBM/adversarial-robustness-toolbox)


# Installation & Configuration
``` 
pip install armory-testbed
```

Upon installing armory, a directory will be created at `~/.armory`. This user 
specific folder is the default directory for downloaded datasets, model weights, and 
evaluation outputs. 

To change these default directories simply run `armory configure` after installation.

# Usage

There are four ways to interact with the armory container system.

1) `armory run <path/to/config.json>`. 
This will run a [configuration file](docs/configuration_files.md) end to end. Stdout 
and stderror logs will be displayed to the user, and the container will be removed 
gracefully upon completion. Results from the evaluation can be found in your output 
directory.

2) `armory launch <tf1|tf2|pytorch> --interactive`. 
This will launch a framework specific container, with appropriate mounted volumes, for 
the user to attach to for debugging purposes. A command to attach to the container will
be returned from this call, and it can be ran in a separate terminal. To later close 
the interactive container simply run CTRL+C from the terminal where this command was 
ran.

3) `armory launch <tf1|tf2|pytorch> --jupyter`. 
Similar to the interactive launch, this will spin up a container for a specific 
framework, but will instead return the web address of a jupyter lab server where 
debugging can be performed. To close the jupyter server simply run CTRL+C from the 
terminal where this command was ran.

4) `armory exec <tf1|tf2|pytorch> -- <cmd>`. 
This will run a specific command within a framework specific container. A notable use
case for this would be to run test cases using pytest. After completion of the command 
the container will be removed.

Note: Since ARMORY launches Docker containers, the python package must be ran on system host.

### Example usage:
```
pip install armory-testbed
armory configure
git clone https://github.com/twosixlabs/armory-example.git
cd armory-example
armory run examples/fgm_attack.json
```

# Scenarios
Armory will provide several scenarios for various data modalities and threat models. 
More information will be added 

# FAQs
Please see the [frequently asked questions](docs/faqs.md) documentation for more information on:
* Datasets returning NumPy arrays
* Access to underlying models from wrapped classifiers.

# Contributing
Armory is an open source project and as such we welcome contributions! Please refer to 
our [contribution docs](docs/contributing.md) for how to get started.
