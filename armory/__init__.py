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

try:
    from importlib.metadata import version, PackageNotFoundError
except ModuleNotFoundError:
    from importlib_metadata import version, PackageNotFoundError

import pathlib
import re
import sys
import subprocess

from armory.logs import log


def get_dynamic_version():
    """
    Produce the version dynamically from setup.py if available.

    Return None if setup.py is not available
    """
    armory_repo_root = pathlib.Path(__file__).parent.parent
    setup = armory_repo_root / "setup.py"
    if not setup.is_file():
        return None

    completed = subprocess.run(
        ["python", str(setup), "--version"],
        cwd=str(armory_repo_root),
        capture_output=True,
        text=True,
    )
    try:
        completed.check_returncode()
    except subprocess.CalledProcessError:
        log.critical("setup.py exists but 'python setup.py --version' failed.")
        raise
    version = completed.stdout.strip()
    return version


__version__ = get_dynamic_version()
if __version__ is None:
    try:
        __version__ = version("armory-testbed")
    except PackageNotFoundError:
        log.critical("armory package is not pip installed and not locally cloned")
        raise
__version__ = re.sub(r"dev\d+\+(g[0-9a-f]+)(\.d\d+)?$", r"\1", __version__)


# If just querying version, stop and exit
if (
    len(sys.argv) == 2
    and (sys.argv[0] == "-m" or pathlib.Path(sys.argv[0]).stem == "armory")
    and sys.argv[1] in ("-v", "--version", "version")
):
    print(f"{__version__}")
    sys.exit(0)

# Handle PyTorch / TensorFlow interplay

# import torch before tensorflow to ensure torch.utils.data.DataLoader can utilize
#     all CPU resources when num_workers > 1
try:
    import torch  # noqa: F401
except ImportError:
    pass

# From: https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
try:
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        log.info("Setting tf.config.experimental.set_memory_growth to True on all GPUs")
except RuntimeError:
    log.exception("Import armory before initializing GPU tensors")
    raise
except ImportError:
    pass

# Handle ART configuration

from armory import paths

try:
    paths.set_art_data_path()
except OSError:
    # If running in --no-docker mode, catch write error based on default DockerPaths
    # the later call to paths.set_mode("host") will set this properly
    pass

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
