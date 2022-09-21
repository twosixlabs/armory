"""
ARMORY Versions use "Semantic Version" scheme where stable releases will have versions
like `0.14.6`.  Armory uses the most recent git tag for versioning. For example if the
most recent git tag is `v0.14.6`, then the version will be `0.14.6`.

If you are a developer, the version will be constructed from the most recent tag plus a
suffix of gHASH where HASH is the short hash of the most recent commit. For example,
if the most recent git tag is v0.14.6 and the most recent commit hash is 1234567 then
the version will be 0.14.6.g1234567. This scheme does differ from the scm strings
which also have a commit count and date in them like 1.0.1.dev2+g0c5ffd9.d20220314181920
which is a bit ungainly.
"""

import re
import shutil
import pathlib
import subprocess

try:
    from importlib import metadata
except (ModuleNotFoundError, ImportError):
    # Python <= 3.7
    import importlib_metadata as metadata  # type: ignore


from armory.logs import log


def make_version_tuple(version_str: str) -> tuple:
    return tuple(map(int, (version_str.split("."))))


def trim_version(version_str) -> str:
    git_tag_regex = re.compile(r"[vV]?(?P<version>\d+(?:\.\d+){0,2})")
    tag_match     = git_tag_regex.match(version_str)
    if tag_match is not None:
        return tag_match.group("version")
    return None


def get_build_hook_version() -> str:
    try:
        from armory.__about__ import version_tuple
        return ".".join(map(str, version_tuple[:3]))
    except Exception as error:
        log.error(f"ERROR: Unable to extract version from __about__.py")
    return None


def get_metadata_version(package: str, version=None) -> str:
    try:
        return trim_version(str(metadata.version(package)))
    except PackageNotFoundError:
        log.error(f"version.py: Unable to find the specified package! Package {package} not installed.")
    return None


def get_tag_version() -> str:
    # See: https://github.com/pypa/setuptools_scm/blob/main/src/setuptools_scm/git.py
    git_dir  = None
    git_describe = [ "git", "describe", "--dirty", "--tags", "--long" ]

    for exec_path in (pathlib.Path(__file__), pathlib.Path.cwd()):
        if pathlib.Path(exec_path / ".git").is_dir():
            git_dir = exec_path
            break

    if git_dir is None or shutil.which('git') is None:
        # Unable to find `.git` directory or git executable
        # is not installed.
        return None

    describe_out = subprocess.run(
        git_describe,
        capture_output=True,
        cwd=str(git_dir),
        text=True,
    ).stdout

    tag_version = trim_version(describe_out)
    if tag_version is not None:
        return tag_version

    return None


def get_version() -> str:
    version = get_metadata_version("armory")
    if version is None:
        version = get_build_hook_version()
    if version is None:
        version = get_tag_version()
    return version or "0.0.0"