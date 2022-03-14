"""
Enables programmatic accessing of most recent docker images
"""

import armory
import pkg_resources

DOCKER_REPOSITORY = "twosixarmory"
ARMORY_VERSION = armory.__version__
TAG = armory.__version__

PYTORCH = f"{DOCKER_REPOSITORY}/pytorch:{TAG}"
PYTORCH_DEEPSPEECH = f"{DOCKER_REPOSITORY}/pytorch-deepspeech:{TAG}"
TF2 = f"{DOCKER_REPOSITORY}/tf2:{TAG}"
ALL = (
    PYTORCH,
    PYTORCH_DEEPSPEECH,
    TF2,
)
REPOSITORIES = tuple(x.split(":")[0] for x in ALL)
IMAGE_MAP = {
    "pytorch": PYTORCH,
    "pytorch-deepspeech": PYTORCH_DEEPSPEECH,
    "tf2": TF2,
}


def parse_version(tag):
    """
    Return PEP 440 version for given version tag
    """
    if not isinstance(tag, str):
        raise ValueError(f"tag is a {type(tag)}, not a str")
    # if len(tag.split(".")) == 4:
    #     log.warning(f"Using Experimental Version of Armory: {tag}")
    #     if tag.split(".")[-1]
    # elif len(tag.split(".")) != 3:
    #     raise ValueError(f"tag {tag} must be of form 'major.minor.patch(.dev...)'")
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
