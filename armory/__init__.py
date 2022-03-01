"""Adversarial Robustness Evaluation Test Bed

ARMORY Versions use "Semantic Version" scheme
where stable releases will have versions like
`0.14.6`.  Armory uses `setuptools_scm` under the
hood which pulls the version from the .git
information.  Stable release will be denoted by
git tags.  If you are a developer, and you modify
armory locally, the version will follow the format
as specified by `setuptools_scm`.

For example: if you checkout a stable release version
of armory (e.g. `git checkout v0.14.5`) and then make
a local edit to a file, the `version` will be something
like `dev142+gc650673.d20220301143014`.
"""

from importlib_metadata import version, PackageNotFoundError
from armory.logs import log

try:
    __version__ = version("armory-testbed")
    log.info(f"Armory Version: {__version__} Installed.")
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
