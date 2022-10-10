Armory Style Guide
=====================
The following describes the stlying approach used during `armory` development.  If you plan to make
contributions to armory, please follow these style guidelines.  Note: Some of these are enforced by
our CI process and we have provided some `git hooks` to help with the formatting.  For more information
see [Pre-commit Hooks](#pre-commit-hooks) below.


# Formatting
All contributions to the repository must be formatted with [black](https://github.com/psf/black).
```
pip install black==22.*
black .
```
We will update black versioning annually following their [Stability Policy](https://black.readthedocs.io/en/stable/the_black_code_style/index.html#stability-policy).

All JSON files committed to the repository must be formatted using the following command:
```
python -m tools.format_json
```
It is based off of Python's [json.tool](https://docs.python.org/3/library/json.html#module-json.tool)
with the `--sort-keys` argument, though overcomes an issue in 3.6 which made it unable to rewrite
the file it was reading from.

We use [Flake8](https://flake8.pycqa.org/) for non-formatting PEP style enforcement.
```
flake8
```
Our repo-specific Flake8 configuration is detailed in `.flake8`.

Our repo uses [yamllint](https://yamllint.readthedocs.io/en/stable/) for enforcement of YAML
syntax and formatting.
```
yamllint --no-warnings
```

Our repo-specific configuration for yamllint is found in `.yamllint`.

### Pre-commit Hooks

If you want those tools to run automatically before each commit, run:
```bash
cat tools/pre-commit.sh > .git/hooks/pre-commit
chmod 755 .git/hooks/pre-commit
```
Note: these hooks depend on some python tools being installed in your environment. These
can be installed using:
```bash
pip install .[developer]
```
For more information about how to contribute to armory, see [Contributing to Armory](docs/CONTRIBUTING.md).


# Import Style
Imports in python files should be organized into three blocks, lexically ordered, after the docstring, and before other code:
* Block 1: built-in package imports
* Block 2: external package imports
* Block 3: internal package imports
These blocks should be separated by a single empty line. Here is an example:
```python
"""
Docstring
"""

import json
import os

import requests
import numpy as np
from art import defences

from armory.docker.management import ManagementInstance
from armory.utils.external_repo import download_and_extract_repos
from armory.logs import log
# ...
```

Exceptions are allowed for import error handling, required import ordering, or in-class/function imports.

## Additional Import Block for Downloaded GitHub Repos

A fourth import block may be added for external package imports that require downloading an external github repo via armory.
This is typically only used for some baseline models and art experimental attacks.
This must use the `armory.errors.ExternalRepoImport` context manager as follows (one `with` statement per external repo):
```
from armory.errors import ExternalRepoImport

with ExternalRepoImport(
    repo="colour-science/colour@v0.3.16",
    experiment="carla_obj_det_dpatch_undefended.json",
):
    import colour
```

`repo` refers to the GitHub repo name (optionally with `@tag`).
`experiment` refers to the `.json` file in the `scenario_config` directory that uses this repo.
These repos are specifically *NOT* installed in the armory-supported docker containers and conda environments, and are downloaded at runtime.
The rationale is that they are not a core part of the library and are meant to mirror usage of individuals evaluating their own models.
