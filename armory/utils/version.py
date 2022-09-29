'''
ARMORY Versions use "Semantic Version" scheme where stable releases will have versions
like `0.14.6`.  Armory uses the most recent git tag for versioning. For example if the
most recent git tag is `v0.14.6`, then the version will be `0.14.6`.

If you are a developer, the version will be constructed from the most recent tag plus a
suffix of gHASH where HASH is the short hash of the most recent commit. For example,
if the most recent git tag is v0.14.6 and the most recent commit hash is 1234567 then
the version will be 0.14.6.g1234567. This scheme does differ from the scm strings
which also have a commit count and date in them like 1.0.1.dev2+g0c5ffd9.d20220314181920
which is a bit ungainly.
'''

import shutil
import subprocess
from pathlib import Path

try:
    from importlib import metadata
except ModuleNotFoundError:
    # Python <= 3.7
    from importlib_metadata import version, PackageNotFoundError  # noqa

from armory.logs import log


def to_docker_tag(version_str: str) -> str:
    '''Convert version string to docker tag'''
    return version_str.replace('+', '.')


def normalize_git_version(git_output: str) -> str:
    '''Normalizes `git describe` output for pip.
    NOTE: This does not add a `+build` tag if pulled from a tagged release.
    '''
    normalized_version = git_output.strip().lstrip('v')
    normalized_version = [part.lstrip('g') for part in normalized_version.split('-')]
    normalized_version = '+build'.join(normalized_version[0::2])
    return normalized_version


def get_build_hook_version(version_str: str = '') -> str:
    '''Retrieve the version from the build hook'''
    try:
        from armory.__about__ import __version__ as version_str
    except ModuleNotFoundError:
        log.error("ERROR: Unable to extract version from __about__.py")
    return version_str


def get_metadata_version(package: str, version_str: str = '') -> str:
    '''Retrieve the version from the package metadata'''
    try:
        return str(metadata.version(package))
    except metadata.PackageNotFoundError:
        log.error(f"ERROR: Unable to find the specified package! Package {package} not installed.")
    return version_str


def get_tag_version(git_dir: Path = None) -> str:
    '''Retrieve the version from the most recent git tag'''
    if shutil.which('git') is None:
        raise RuntimeError('git is not installed')
    git_describe = subprocess.run(
        ['git', 'describe', '--tags'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    if bool(git_describe.returncode) or not bool(git_describe.stdout):
        raise RuntimeError('Unable to retrieve git tag')
    return normalize_git_version(git_describe.stdout.decode('utf-8'))


def get_version(package_name: str = 'armory-testbed', version_str: str = '') -> str:
    version_str = get_metadata_version(package_name)
    if not bool(version_str):
        version_str = get_build_hook_version()
    if not bool(version_str):
        version_str = get_tag_version()
    return version_str or "0.0.0"
