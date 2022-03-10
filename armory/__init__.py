"""Adversarial Robustness Evaluation Test Bed

ARMORY Versions use "Semantic Version" scheme where stable releases will have versions
like `0.14.6`.  Armory uses `setuptools_scm` which pulls the version from the
tags most recent git tag. For example if the most recent git tag is `v0.14.6`,
then the version will be `0.14.6`.

If you are a developer, the version will be constructed from the most recent tag
plus a suffix of devN-gHASH where N is the number of commits since the most recent
tag and HASH is the short hash of the most recent commit. For example, if the most
recent git tag is v0.14.6 and the most recent commit hash is 1234567 then the
built image tage will be v0.14.6-dev1-g1234567.
"""

from importlib_metadata import version, PackageNotFoundError
from armory.logs import log

try:
    __version__ = version("armory-testbed")
except PackageNotFoundError as e:
    log.error("Armory Package is not Installed...")
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
