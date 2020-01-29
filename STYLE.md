# Formatting
All contributions to the repository must be formatted with [black](https://github.com/psf/black).
```
black .
```

All JSON files committed to the repository must be formatted using the following command:
```
python -m scripts.format_json
```
It is based off of Python's [json.tool](https://docs.python.org/3/library/json.html#module-json.tool)
with the `--sort-keys` argument, though overcomes an issue in 3.6 which made it unable to rewrite
the file it was reading from.

# Import Style
Imports in python files should be organized into three blocks, after the docstring, and before other code:
* Block 1: built-in package imports
* Block 2: external package imports
* Block 3: internal package imports
These blocks should be separated by a single empty line. Here is an example:
```python
"""
Docstring
"""

import json
import logging
import os

import requests
import numpy as np
from art import defences

from armory.data.common import SUPPORTED_DATASETS                                         
from armory.docker.management import ManagementInstance                                   
from armory.utils.external_repo import download_and_extract


logger = logging.getLogger(__name__)
# ...
```

Exceptions are allowed for import error handling, required import ordering, or in-class/function imports.

