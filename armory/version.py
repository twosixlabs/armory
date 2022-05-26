"""
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
import subprocess

from armory.logs import log


_VERSION = None


def get_version():
    global _VERSION
    if _VERSION is None:
        _VERSION = generate_version()
    return _VERSION


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
    _version = completed.stdout.strip()
    return _version


def get_pip_version():
    try:
        return version("armory-testbed")
    except PackageNotFoundError:
        log.critical("armory package is not pip installed and not locally cloned")
        raise


def trim_version(_version):
    return re.sub(r"dev\d+\+(g[0-9a-f]+)(\.d\d+)?$", r"\1", _version)


def generate_version():
    _version = get_dynamic_version()
    if _version is None:
        _version = get_pip_version()
    _version = trim_version(_version)
    return _version
