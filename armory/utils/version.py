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

import os
import re
import site
import setuptools_scm

from pathlib import Path

try:
    from importlib import metadata
except ModuleNotFoundError:
    # Python <= 3.7
    import importlib_metadata as metadata # noqa

from armory.logs import log


def to_docker_tag(version_str: str) -> str:
    '''Convert version string to docker tag'''
    return version_str.replace('+', '.')


def get_metadata_version(package: str, version_str: str = '') -> str:
    '''Retrieve the version from the package metadata'''
    try:
        return str(metadata.version(package))
    except metadata.PackageNotFoundError:
        log.error(f"ERROR: Unable to find the specified package! Package {package} not installed.")
    return version_str


def get_build_hook_version(version_str: str = '') -> str:
    '''Retrieve the version from the build hook'''
    try:
        from armory.__about__ import __version__ as version_str
    except ModuleNotFoundError:
        log.error("ERROR: Unable to extract version from __about__.py")
    return version_str


def get_tag_version(git_dir: Path = None) -> str:
    '''Retrieve the version from the most recent git tag'''
    project_paths = [Path(__file__).parent.parent, Path.cwd()]
    git_dir = list(filter(lambda path: Path(path / ".git").is_dir(), project_paths))
    scm_config = {
        'root': git_dir,
        'relative_to': __file__,
        'version_scheme': "post-release",
        'local_scheme': "node-and-date",
    }
    if not git_dir:
        log.error("ERROR: Unable to find `.git` directory!")
        return
    scm_config.update({'root': git_dir[0]})
    return setuptools_scm.get_version(**scm_config).replace("+", ".")


def developer_mode_version(
        package_name: str,
        pretend_version: str = False,
        update_metadata: bool = False) -> str:
    '''Return the version in developer mode

    Args:
        param1 (int): The first parameter.
        package_name (str): The name of the package.
        pretend_version (str): The version to pretend to be.
        update_metadata (bool): Whether to update the metadata.

    Example:
        $ ARMORY_DEV_MODE=1 ARMORY_PRETEND_VERSION="1.2.3" armory --version
    '''
    old_version = get_metadata_version(package_name)
    version_str = pretend_version or get_tag_version()

    if pretend_version:
        log.info(f'Spoofing version {pretend_version} for {package_name}')
        return version_str
    if update_metadata:
        version_regex = r'(?P<prefix>^Version: )(?P<version>.*)$'
        [package_meta] = [f for f in metadata.files(package_name) if str(f).endswith('METADATA')] or False
        if not package_meta:
            log.error(f'Unable to find package metadata for {package_name}')
            return version_str
        for path in site.getsitepackages():
            metadata_path = Path(path / package_meta)
            if metadata_path.is_file():
                break
        metadata_update = re.sub(
            version_regex,
            f'\g<prefix>{version_str}',  # noqa
            metadata_path.read_text(),
            flags=re.M)
        metadata_path.write_text(metadata_update)
        log.info(f'Version updated from {old_version} to {version_str}')

    return version_str


def get_version(package_name: str = 'armory-testbed', version_str: str = '') -> str:
    if os.getenv('ARMORY_DEV_MODE'):
        pretend_version = os.getenv('ARMORY_PRETEND_VERSION')
        update_metadata = os.getenv('ARMORY_UPDATE_METADATA')
        return developer_mode_version(package_name, pretend_version, update_metadata)

    version_str = get_build_hook_version()
    if not version_str:
        version_str = get_metadata_version(package_name)
    if not version_str:
        version_str = get_tag_version()
    return version_str
