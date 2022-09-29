#!/usr/bin/env python
"""Retrieve's project version during installation using git tags and commit hash.

The primary objective of this hook are to generate a pip compliant(e.g. pip_version)
canonical version string from git tags and commit hash. This also servers as the base
mechanism for ensuring compliance between the `pip_version` and an acceptable docker
tag(e.g. `docker_version`).

EXAMPLE:
  >>> docker_version = pip_version.replace('+', '.')

RESOURCES:
  - https://peps.python.org/pep-0440/
  - https://github.com/pypa/setuptools_scm/blob/main/src/setuptools_scm/version.py#L34
  - https://github.com/pypa/setuptools_scm/blob/main/src/setuptools_scm/git.py
"""
import os
import shutil
import contextlib
import subprocess
import setuptools

from pathlib import Path
from packaging.version import Version


SOURCE_DIR = 'armory'
VERSION_FILE = '__about__.py'
VERSION_FILE_TEMPLATE = """\
# coding: utf-8
# File automatically generated during installation.
# Do not change or track in version control.
__version__ = {version!r}
"""

GIT_DIR_CMD = ['git', 'rev-parse', '--git-dir']
GIT_DESCRIBE_CMD = ['git', 'describe', '--tags']


@contextlib.contextmanager
def working_directory(path):
    """Changes working directory and returns to previous on exit."""
    prev_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)


def execute_command(command: list, cwd: Path = None) -> str:
    """Execute a command and return the output."""
    # TODO: Return (exitcode, stdout, stderr) tuple
    def command_runner(command:list) -> str:
        return subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        ).stdout.decode('utf-8').strip()

    if cwd is not None:
        with working_directory(cwd):
            return command_runner(command)
    return command_runner(command)


def normalize_version(git_output: str) -> str:
    """Normalize `git describe` output.
    NOTE: This does not add a `+build` tag if pulled from a tagged release.
    """
    normalized_version = git_output[1:] if git_output.startswith('v') else git_output
    normalized_version = [part.lstrip('g') for part in normalized_version.split('-')]
    normalized_version = '+build'.join(normalized_version[0::2])
    return normalized_version


def get_version(git_dir: Path = None) -> str:
    """Retrieve the version from git."""
    git_describe = execute_command(GIT_DESCRIBE_CMD, cwd=git_dir)
    version_string = normalize_version(git_describe)
    # Check if the version is PEP 440 compliant
    try:
        version_string = str(Version(version_string))
    except ValueError:
        raise ValueError(f'Version {version_string} is not PEP 440 compliant')
    return version_string


def version_hook():
    if shutil.which('git') is None:
        raise RuntimeError('git is not installed')
    git_dir = Path(execute_command(GIT_DIR_CMD)).absolute()
    version_file = Path((git_dir.parent) / SOURCE_DIR / VERSION_FILE)
    version_string = get_version(git_dir)
    # Write version to file. This is useful as a durable source of truth for the version.
    version_file.write_text(VERSION_FILE_TEMPLATE.format(version=version_string))
    version_file.chmod(0o644)
    return version_string


# This is used by `hatch` to determine the version.
# See: https://hatch.pypa.io/latest/plugins/version-source/code/
__version__ = version_hook()


if __name__ == "__main__":
    # Needed for GitHubs dependency graph.
    setuptools.setup()
