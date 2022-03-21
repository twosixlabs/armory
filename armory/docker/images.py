"""
Enables programmatic accessing of most recent docker images
"""

import docker
import docker.errors
import requests

import armory
from armory.logs import log
from armory.utils import docker_api

TAG = armory.__version__
log.trace(f"armory.__version__: {armory.__version__}")

DOCKER_REPOSITORY = "twosixarmory"

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


# TODO is_old is used by the two implementations of `armory clean` in clean()
# with the advent of setuptools_scm, this no longer makes sense. Since
# I don't know if `armory clean` is used anywhere else, NotImplemented is a rude
# way to find out
def is_old(tag: str):
    raise NotImplementedError


def split_name(image_name: str):
    """
    Parse image name and return tuple (repo, name, tag)
        Return the empty string "" for any that do not exist
    """
    if not image_name:
        raise ValueError("image_name cannot be empty")

    tokens = image_name.split(":")
    if len(tokens) == 1:
        repo_name = tokens[0]
        tag = ""
    elif len(tokens) == 2:
        repo_name, tag = tokens
        if "/" in tag:
            raise ValueError(
                f"invalid docker image image_name {image_name}. '/' cannot be in tag"
            )
    else:
        raise ValueError(f"invalid docker image image_name {image_name}. Too many ':'")

    if "/" in repo_name:
        i = repo_name.rindex("/")
        repo = repo_name[:i]
        name = repo_name[i + 1 :]
    else:
        repo = ""
        name = repo_name

    return repo, name, tag


def join_name(repo: str, name: str, tag: str):
    """
    Inverse of split_image_name. The following should be identity operations:
        image_name = join_name(*split_name(image_name))
        repo, name, tag = split_name(join_name(repo, name, tag))
    """
    if not name:
        raise ValueError("name cannot be empty")
    if repo:
        repo = repo + "/"
    if tag:
        tag = ":" + tag
    return f"{repo}{name}{tag}"


def is_armory(image_name: str):
    """
    Takes a docker image name, returns False if not a armory image
        If an armory image, it will return `twosixarmory/<framework>:tag`
    """
    repo, name, _ = split_name(image_name)
    if repo and repo != DOCKER_REPOSITORY:
        return False
    return name in IMAGE_MAP


def ensure_image_present(self, image_name: str) -> str:
    """
    If image_name is available, return it. Otherwise, pull it from dockerhub.
    """
    log.trace(f"ensure_image_present {image_name}")
    docker_client = docker.from_env()

    if not is_armory(image_name):
        log.trace(f"asking local docker for image {image_name}")
        try:
            docker_client.images.get(image_name)
            log.success(f"found docker image {image_name}")
            return image_name
        except docker.errors.ImageNotFound:
            log.trace(f"image {image_name} not found")
        except requests.exceptions.HTTPError:
            log.trace(f"http error when looking for image {image_name}")
            raise

        try:
            docker_api.pull_verbose(docker_client, image_name)
            return image_name
        except docker.errors.NotFound:
            log.error(f"Image {image_name} could not be downloaded")
            raise
        except requests.exceptions.ConnectionError:
            log.error("Docker connection refused. Is Docker Daemon running?")
            raise

    #

    # look first for the versioned and then the unversioned, return if hit
    # if there is a tag present, use that. otherwise add the current version
    if ":" in image_name:
        checks = (image_name,)
    else:
        # TODO: This needs to be fixed if 'image_name' does not refer to twosixarmory image
        #   There should be a more explicit check for specific armory image names.
        check = f"{image_name}:{armory.__version__}"
        check_previous = ".".join(check.split(".")[:3])
        if check_previous != check:
            checks = (check, check_previous)
        else:
            checks = (check,)

    for check in checks:
        log.trace(f"asking local docker for image {check}")
        try:
            docker_client.images.get(check)
            log.success(f"found docker image {image_name} as {check}")
            return check
        except docker.errors.ImageNotFound:
            log.trace(f"image {check} not found")
        except requests.exceptions.HTTPError:
            log.trace(f"http error when looking for image {check}")
            raise

    log.info(f"image {image_name} not found. downloading...")
    try:
        docker_api.pull_verbose(docker_client, image_name)
        return image_name
    except docker.errors.NotFound:
        if image_name in ALL:
            raise ValueError(
                "You are attempting to pull an unpublished armory docker image.\n"
                "This is likely because you're running armory from a dev branch. "
                "If you want a stable release with "
                "published docker images try pip installing 'armory-testbed' "
                "or using out one of the release branches on the git repository. "
                "If you'd like to continue working on the developer image please "
                "build it from source on your machine as described here:\n"
                "https://armory.readthedocs.io/en/latest/contributing/#development-docker-containers\n"
                "python docker/build.py --help"
            )
        else:
            log.error(f"Image {image_name} could not be downloaded")
            raise
    except requests.exceptions.ConnectionError:
        log.error("Docker connection refused. Is Docker Daemon running?")
        raise
