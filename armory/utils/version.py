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

import os
import re
import site
import functools
import setuptools_scm

from pathlib import Path
from types import FunctionType

try:
    from importlib import metadata
except ImportError:
    # Python <= 3.7
    import importlib_metadata as metadata  # noqa

from armory.logs import log


class VersionError(Exception):
    pass


def to_docker_tag(version_str: str) -> str:
    """Convert version string to docker tag"""
    return version_str.replace("+", ".")


def version_resource() -> str:
    checks = [
        get_tag_version,
        get_build_version,
        get_pip_version
    ]
    for i, check in enumerate(checks):
        if isinstance(check, FunctionType):
            try:
                return check()
            except Exception as e:
                checks[i] = e
    for check in checks:
        if isinstance(check, Exception):
            log.warning(f"Unable to retrieve version: {check}")
    raise RuntimeError("Unable to determine version number!")


def get_pip_version() -> str:
    """Retrieve the version from the package metadata"""
    try:
        return str(metadata.version("armory-testbed"))
    except metadata.PackageNotFoundError:
        raise VersionError(f"ERROR: Unable to find Armory package! Armory is not installed.")


def get_build_version(version_str: str = "") -> str:
    """Retrieve the version from the build hook"""
    try:
        from armory.__about__ import __version__ as version_str
    except ModuleNotFoundError:
        raise VersionError("Unable to extract version from __about__.py")
    return version_str


def get_tag_version() -> str:
    """Retrieve the version from the most recent git tag"""
    project_root = Path(__file__).parent.parent.parent
    scm_config = {
        "root": project_root,
        "version_scheme": "post-release",
    }
    if Path(project_root / ".git").is_dir():
        return setuptools_scm.get_version(**scm_config)
    raise VersionError("ERROR: Unable to find `.git` directory!")


def developer_mode_version(
    package_name: str, pretend_version: str = "", update_metadata: bool = False
) -> str:
    """Return the version in developer mode

    Args:
        package_name (str): The name of the package.
        pretend_version (str): The version to pretend to be.
        update_metadata (bool): Whether to update the metadata.

    Example:
        $ ARMORY_DEV_MODE=1 ARMORY_PRETEND_VERSION="1.2.3" armory --version
    """
    old_version = get_pip_version(package_name)
    version_str = pretend_version or get_tag_version()
    if pretend_version:
        log.info(f"Spoofing version {pretend_version} for {package_name}")
    if update_metadata:
        version_regex = r"(?P<prefix>^Version: )(?P<version>.*)$"
        package_meta = None
        for f in metadata.files(package_name):
            if str(f).endswith("METADATA"):
                package_meta = f
                break
        if not package_meta:
            log.warning(f"Unable to find package metadata for {package_name}")
            return version_str
        for path in site.getsitepackages():
            metadata_path = Path(path / package_meta)
            if metadata_path.is_file():
                break
        metadata_update = re.sub(
            version_regex,
            f"\g<prefix>{version_str}",  # noqa
            metadata_path.read_text(),
            flags=re.M,
        )
        metadata_path.write_text(metadata_update)
        log.info(f"Version updated from {old_version} to {version_str}")
    return version_str


@functools.lru_cache(maxsize=1, typed=False)
def get_version() -> str:
    if os.getenv("ARMORY_DEV_MODE"):
        pretend_version = os.getenv("ARMORY_PRETEND_VERSION")
        update_metadata = os.getenv("ARMORY_UPDATE_METADATA")
        return developer_mode_version("armory-testbed", pretend_version, update_metadata)

    return version_resource()
