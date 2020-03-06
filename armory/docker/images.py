"""
Enables programmatic accessing of most recent docker images
"""

from distutils import version

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
REPOSITORIES = tuple(x.split(":")[0] for x in ALL)


def dev_version(tag):
    """
    Return version.LooseVersion class for given version tag
    """
    if not tag.endswith(armory.DEV):
        raise ValueError(f"invalid dev tag: {tag} does not end with {armory.DEV}")

    # check that the remaining version number is strict
    strict = tag[: -len(armory.DEV)]
    version.StrictVersion(strict)
    return version.LooseVersion(tag)


if TAG.endswith(armory.DEV):
    VERSION = dev_version(TAG)
else:
    VERSION = version.StrictVersion(TAG)


def is_old(tag: str):
    """
    Return True if tag is an old armory container, False otherwise

    If current version is dev, only returns True for old "-dev" containers.
    """
    if armory.is_dev():
        raise NotImplementedError("dev tag")

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
            if armory.is_dev():
                if not tag.endswith(armory.DEV):
                    return False
                tag = dev_version(tag)
                if tag < VERSION:
                    return True
            elif version.StrictVersion(tag) < VERSION:
                return True
        except (AttributeError, ValueError):
            # Catch empty tag and tag parsing errors
            pass
    return False
