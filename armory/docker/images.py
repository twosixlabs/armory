"""
Enables programmatic accessing of most recent docker images
"""

import pkg_resources

import armory

USER = "twosixarmory"
TAG = armory.__version__

PYTORCH = f"{USER}/pytorch:{TAG}"
PYTORCH_DEEPSPEECH = f"{USER}/pytorch-deepspeech:{TAG}"
TF1 = f"{USER}/tf1:{TAG}"
TF2 = f"{USER}/tf2:{TAG}"
ALL = (
    PYTORCH,
    PYTORCH_DEEPSPEECH,
    TF1,
    TF2,
)
REPOSITORIES = tuple(x.split(":")[0] for x in ALL)
IMAGE_MAP = {
    "pytorch": PYTORCH,
    "pytorch-deepspeech": PYTORCH_DEEPSPEECH,
    "tf1": TF1,
    "tf2": TF2,
}


def parse_version(tag):
    """
    Return PEP 440 version for given version tag
    """
    if not isinstance(tag, str):
        raise ValueError(f"tag is a {type(tag)}, not a str")
    if len(tag.split(".")) != 3:
        raise ValueError(f"tag {tag} must be of form 'major.minor.patch'")
    version = pkg_resources.parse_version(tag)
    if not isinstance(version, pkg_resources.extern.packaging.version.Version):
        raise ValueError(f"tag {tag} parses to type {type(version)}, not Version")
    return version


VERSION = parse_version(armory.__version__)


def is_old(tag: str):
    """
    Return True if tag is an old armory container, False otherwise
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
