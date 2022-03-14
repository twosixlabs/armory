"""Adversarial Robustness Evaluation Test Bed

ARMORY Versions use "Semantic Version" scheme where stable releases will have versions
like `0.14.6`.  Armory uses `setuptools_scm` which pulls the version from the tags most
recent git tag. For example if the most recent git tag is `v0.14.6`, then the version
will be `0.14.6`.

If you are a developer, the version will be constructed from the most recent tag plus a
suffix of gHASH where HASH is the short hash of the most recent commit. For example,
if the most recent git tag is v0.14.6 and the most recent commit hash is 1234567 then
the version will be 0.14.6.g1234567. This scheme does differ from the scm strings
which also have a commit count and date in them like 1.0.1.dev2+g0c5ffd9.d20220314181920
which is a bit ungainly.
"""

from importlib_metadata import version, PackageNotFoundError
from armory.logs import log
import re

try:
    __version__ = version("armory-testbed")
    __version__ = re.sub(r"dev\d+\+(g[0-9a-f]+)\.d\d+$", r"\1", __version__)
except PackageNotFoundError as e:
    log.crticial("armory package is not installed")
    raise e

# typedef for a widely used JSON-like configuration specification
from typing import Dict, Any

Config = Dict[str, Any]

# Submodule imports
try:
    from armory import art_experimental
    from armory import baseline_models
    from armory import data
    from armory import docker
    from armory import eval
    from armory import paths
    from armory import utils
    from armory import webapi
except ImportError as e:
    module = e.name
    print(f"ERROR: cannot import '{module}' module")
    print("    Please run: $ pip install -r requirements.txt")
    raise

END_SENTINEL = "Scenario has finished running cleanly"
