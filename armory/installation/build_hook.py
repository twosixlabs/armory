'''Retrieve's project version during installation using git tags and commit hash.

The primary objective of this hook are to generate a pip compliant(e.g. pip_version)
canonical version string from git tags and commit hash. This also servers as the base
mechanism for ensuring compliance between the `pip_version` and an acceptable docker
tag (e.g. `docker_version`).

EXAMPLE:
  >>> docker_version = pip_version.replace('+', '.')

RESOURCES:
  - https://peps.python.org/pep-0440/
  - https://github.com/pypa/setuptools_scm/blob/main/src/setuptools_scm/version.py#L34
  - https://github.com/pypa/setuptools_scm/blob/main/src/setuptools_scm/git.py
'''

import os
import shutil
import contextlib
import subprocess

from pathlib import Path
from packaging.version import Version


GIT_TOPLEVEL_COMMAND = ""
GIT_DIR_COMMAND = ""
# GIT_DESCRIBE = execute_command(['git', 'describe', '--tags'])
VERSION_FILE_TEMPLATE = ""


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
  """Normalize `git describe` output."""
  # TODO: While this does not add a `+build` tag if pulled from a tagged release,
  #       more checks should take place to ensure that the version is PEP 440 compliant.
  normalized_version = git_output[1:] if git_output.startswith('v') else git_output
  normalized_version = [part.lstrip('g') for part in normalized_version.split('-')]
  normalized_version = '+build'.join(normalized_version[0::2])
  return normalized_version


def write_version_file(version_string: str, version_file: str, git_dir: Path = None):
  """Write the version string to a file.
  NOTE:
    This is useful as a durable source of truth for the version.
  """
  repo_root = Path(git_dir).parent
  version_file = Path(repo_root / "armory" / version_file)
  version_template = f""
  print(version_file)
  print(Path(__file__))
  print(Path(git_dir).parent)
  #.relative_to(__file__))
  ...



def version_hook():
  # TODO: Add a mechanism to pull version information w/out the use of `git`.
  if shutil.which('git') is None:
    raise RuntimeError('git is not installed')

  # Locate the git directory
  git_dir = execute_command(['git', 'rev-parse', '--git-dir'])
  # TODO: Look into using cwd() along with `git rev-parse --show-toplevel`
  # print(Path().cwd().relative_to(git_dir))

  git_describe = execute_command(['git', 'describe', '--tags'], cwd=git_dir)

  # Normalize the git describe output
  version_string = normalize_version(git_describe)

  # Check if the version is PEP 440 compliant
  try:
    version_check = Version(version_string)
  except ValueError:
    raise ValueError(f'Version {version_string} is not PEP 440 compliant')

  # TODO: Write version to file
  # TODO: Try writting this to about.py and add a path to the pyproject.toml
  write_version_file(version_string, "__about__.py", git_dir)

  print(version_string)
  return version_string


__version__ = version_hook()

# TODO: Check how docker/images handles `+` characters in version strings


