<a name="readme-top"></a>


<!-- PROJECT SHIELDS -->
<!--
  ** Shields utilize markdown "reference style" links for readability.
  ** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
  ** See the bottom of this document for the declaration of the reference variables.
  ** https://www.markdownguide.org/basic-syntax/#reference-style-links
  ** -->
[![CI][ci-badge]][ci-url]
[![PyPI Status Badge][pypi-badge]][pypi-url]
[![PyPI - Python Version][python-badge]][python-url]
[![License: MIT][license-badge]][license-url]
[![Docs][docs-badge]][docs-url]
[![Code style: black][style-badge]][style-url]


<!-- PROJECT LOGO !-->
<br />
<div align="center">
  <a href="/">
    <img src="/docs/assets/logo.png" width="50%" title="ARMORY Logo">
  </a>

  <h3 align="center">Armory</h3>

  <p align="center">
    Scalable test bed for evaluating machine learning model adversarial defenses.
    <br />
    <a href="/docs"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="/issues">Report Bug</a>
    ·
    <a href="/issues">Request Feature</a>
  </p>

</div>



# Overview

ARMORY is a test bed for running scalable evaluations of adversarial defenses.
Configuration files are used to launch local or cloud instances of the ARMORY docker
containers. Models, datasets, and evaluation scripts can be pulled from external
repositories or from the baselines within this project.

Our evaluations are created so that attacks and defenses may be
interchanged. To do this we standardize all attacks and defenses as subclasses of
their respective implementations in the [Adversarial Robustness Toolbox (ART)](https://github.com/Trusted-AI/adversarial-robustness-toolbox) hosted by the LF AI & Data Foundation (LFAI).


# Installation & Configuration
```
pip install armory-testbed
```

Upon installing armory, a directory will be created at `~/.armory`. This user
specific folder is the default directory for downloaded datasets, model weights, and
evaluation outputs.

To change these default directories simply run `armory configure` after installation.

# Usage

There are three ways to interact with the armory container system.

1) `armory run`
* `armory run <path/to/config.json>`.
This will run a [configuration file](/docs/configuration_files.md) end to end. Stdout
and stderror logs will be displayed to the user, and the container will be removed
gracefully upon completion. Results from the evaluation can be found in your output
directory.

* `armory run <path/to/config.json> --interactive`.
This will launch the framework-specific container specified in the
configuration file, copy the configuration file into the container, and provide
the commands to attach to the container in a separate terminal and run the
configuration file end to end while attached to the container. A notable use
case for this would be to debug using pdb. Similar to non-interactive mode, results
from the evaluation can be found in the output directory. To later close the
interactive container simply run CTRL+C from the terminal where this command was ran.

2) `armory launch`
* `armory launch <tf1|tf2|pytorch> --interactive`.
This will launch a framework specific container, with appropriate mounted volumes, for
the user to attach to for debugging purposes. A command to attach to the container will
be returned from this call, and it can be ran in a separate terminal. To later close
the interactive container simply run CTRL+C from the terminal where this command was
ran.

* `armory launch <tf1|tf2|pytorch> --jupyter`.
Similar to the interactive launch, this will spin up a container for a specific
framework, but will instead return the web address of a jupyter lab server where
debugging can be performed. To close the jupyter server simply run CTRL+C from the
terminal where this command was ran.

3) `armory exec`
* `armory exec <tf1|tf2|pytorch> -- <cmd>`.
This will run a specific command within a framework specific container. A notable use
case for this would be to run test cases using pytest. After completion of the command
the container will be removed.

Note: Since ARMORY launches Docker containers, the python package must be ran on
system host (i.e. not inside of a docker container).

### Example usage:
```
pip install armory-testbed
armory configure

git clone https://github.com/twosixlabs/armory-example.git
cd armory-example
armory run official_scenario_configs/cifar10_baseline.json
```

### What is available in the container:
All containers have a pre-installed armory package installed so that baseline models,
datasets, and scenarios can be utilized.

Additionally, volumes (such as your current working directory) will be mounted from
your system host so that you can modify code to be ran, and retrieve outputs.
For more information on these mounts, please see our [Docker documentation](/docs/docker.md#docker-volume-mounts)

# Scenarios
Armory provides several baseline threat-model scenarios for various data modalities.
When running an armory configuration file, the robustness of a defense will be
evaluated against that given scenario. For more information please see our
[Scenario Documentation](/docs/scenarios.md).

# FAQs
Please see the [frequently asked questions](/docs/faqs.md) documentation for more information on:
* Dataset format and preprocessing
* Access to underlying models from wrapped classifiers.

# Contributing
Armory is an open source project and as such we welcome contributions! Please refer to
our [contribution docs](/.github/CONTRIBUTING.md) for how to get started.

# Acknowledgment
This material is based upon work supported by the Defense Advanced Research Projects
Agency (DARPA) under Contract No. HR001120C0114. Any opinions, findings and
conclusions or recommendations expressed in this material are those of the author(s)
and do not necessarily reflect the views of the Defense Advanced Research Projects
Agency (DARPA).


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[ci-badge]: /../../workflows/GitHub%20CI/badge.svg
[ci-url]: /../../actions/workflows/
[pypi-badge]: https://badge.fury.io/py/armory-testbed.svg
[pypi-url]: https://pypi.org/project/armory-testbed
[python-badge]: https://img.shields.io/pypi/pyversions/armory-testbed
[python-url]: https://pypi.org/project/armory-testbed
[license-badge]: https://img.shields.io/badge/License-MIT-yellow.svg
[license-url]: https://opensource.org/licenses/MIT
[docs-badge]: https://readthedocs.org/projects/armory/badge/
[docs-url]: https://readthedocs.org/projects/armory/
[style-badge]: https://img.shields.io/badge/code%20style-black-000000.svg
[style-url]: https://github.com/ambv/black
