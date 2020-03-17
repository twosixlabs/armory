"""
Enables programmatic accessing of most recent docker images
"""

import pkg_resources

import armory

USER = "twosixarmory"
TAG = armory.__version__
TF1 = f"{USER}/tf1:{TAG}"
TF2 = f"{USER}/tf2:{TAG}"
PYTORCH = f"{USER}/pytorch:{TAG}"
ALL = (
    TF1,
    TF2,
    PYTORCH,
)
ARMORY_BASE = f"{USER}/armory:{TAG}"
TF1_BASE = f"{USER}/tf1-base:{TAG}"
TF2_BASE = f"{USER}/tf2-base:{TAG}"
PYTORCH_BASE = f"{USER}/pytorch-base:{TAG}"
BASES = (
    ARMORY_BASE,
    TF1_BASE,
    TF2_BASE,
    PYTORCH_BASE,
)
REPOSITORIES = tuple(x.split(":")[0] for x in (ALL + BASES))


def parse_version(tag):
    """
    Return PEP 440 version for given version tag
    """
    if not isinstance(tag, str):
        raise ValueError(f"tag is a {type(tag)}, not a str")
    if tag.endswith(armory.DEV):
        numeric_tag = tag[: -len(armory.DEV)]
    else:
        numeric_tag = tag
    if len(numeric_tag.split(".")) != 3:
        raise ValueError(f"tag {tag} must be of form 'major.minor.patch[-dev]'")
    version = pkg_resources.parse_version(tag)
    if not isinstance(version, pkg_resources.extern.packaging.version.Version):
        raise ValueError(f"tag {tag} parses to type {type(version)}, not Version")
    return version


VERSION = parse_version(armory.__version__)


def is_old(tag: str):
    """
    Return True if tag is an old armory container, False otherwise

    If current version is dev, only returns True for old "-dev" containers.
    """
    if not isinstance(tag, str):
        raise ValueError(f"tag must be of type str, not type {type(tag)}")
    if tag in ALL:
        return False
    tokens = tag.split(":")
    if len(tokens) != 2:
        return False
    repo, tag = tokens
    if repo in REPOSITORIES:
        try:
            other = parse_version(tag)
            if other < VERSION:
                # return True if both prerelease or both not prerelease
                return not (other.is_prerelease ^ VERSION.is_prerelease)
        except (AttributeError, ValueError):
            # Catch empty tag and tag parsing errors
            pass
    return False
