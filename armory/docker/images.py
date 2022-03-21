"""
Enables programmatic accessing of most recent docker images
"""

import docker
import docker.errors
import requests

import armory
from armory.logs import log, is_progress

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
    if not name:
        raise ValueError(f"invalid docker image image_name {image_name}. No name")

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
    Return whether image_name refers to an armory docker image
    """
    repo, name, _ = split_name(image_name)
    if repo and repo != DOCKER_REPOSITORY:
        return False
    return name in IMAGE_MAP


def get_armory_name(image_name: str):
    """
    Takes a docker image name, returns False if not a armory image
        If an armory image, it will return `twosixarmory/<framework>:tag`
    """
    if not is_armory(image_name):
        raise ValueError(f"Not an armory image name: {image_name}")
    repo, name, tag = split_name(image_name)
    repo = "twosixarmory"
    if tag:  # tag is explicitly defined, use it
        return join_name(repo, name, tag)
    return IMAGE_MAP[name]


def last_armory_release(image_name: str):
    """
    Return the image_name corresponding to the last armory major.minor.patch release
        If the current image_name is a release, return the current
    """
    repo, name, tag = split_name(image_name)
    if not repo or not tag:
        raise ValueError("Must be a full repo/name:tag docker image_name")
    tokens = tag.split(".")
    if len(tokens) == 3:
        return image_name
    elif len(tokens) == 4:
        # remove hash and decrement patch
        major, minor, patch, _ = tokens
        patch = int(patch)
        if patch == 0:
            raise ValueError(f"Tag {tag}: patch cannot be 0 for SCM with hash appended")
        patch -= 1
        patch = str(patch)
        release_tag = ".".join([major, minor, patch])
        return join_name(repo, name, release_tag)
    else:
        raise ValueError(
            f"Tag {tag} must be in major.minor.patch[.hash] SCM version format"
        )


def ensure_image_present(image_name: str) -> str:
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
            pull_verbose(docker_client, image_name)
            return image_name
        except docker.errors.NotFound:
            log.error(f"Image {image_name} could not be downloaded")
            raise
        except requests.exceptions.ConnectionError:
            log.error("Docker connection refused. Is Docker Daemon running?")
            raise

    canon_image_name = get_armory_name(image_name)
    log.info(f"Retrieved canonical image name for {image_name} as {canon_image_name}")

    prev_release = last_armory_release(canon_image_name)
    if canon_image_name != prev_release:  # currently on hashed dev branch
        log.trace(f"asking local docker for image {canon_image_name}")
        try:
            docker_client.images.get(canon_image_name)
            log.success(f"found docker image {image_name} as {canon_image_name}")
            return canon_image_name
        except docker.errors.ImageNotFound:
            log.trace(f"image {canon_image_name} not found")
            log.info(f"reverting to previous release tag image {prev_release}")
        except requests.exceptions.HTTPError:
            log.trace(f"http error when looking for image {canon_image_name}")
            raise

    log.trace(f"asking local docker for image {prev_release}")
    try:
        docker_client.images.get(prev_release)
        log.success(f"found docker image {image_name} as {prev_release}")
        return prev_release
    except docker.errors.ImageNotFound:
        log.trace(f"image {prev_release} not found")
    except requests.exceptions.HTTPError:
        log.trace(f"http error when looking for image {prev_release}")
        raise

    log.info(f"image {prev_release} not found. downloading...")
    try:
        pull_verbose(docker_client, prev_release)
        return prev_release
    except docker.errors.NotFound:
        log.error(f"Image {prev_release} could not be downloaded")
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
    except requests.exceptions.ConnectionError:
        log.error("Docker connection refused. Is Docker Daemon running?")
        raise


def pull_verbose(docker_client, repository, tag=None):
    """
    Use low-level docker-py API to show status while pulling docker containers.
        Attempts to replicate docker command line output if we are showing progress.
    """
    if not is_progress():
        log.info(
            f"docker pulling from {repository}:{tag} use '--log=progress' to see status"
        )
        docker_client.api.pull(repository, tag=tag, stream=False)
        log.success(f"pulled {repository}:{tag}")
        return

    for update in docker_client.api.pull(repository, tag=tag, stream=True, decode=True):
        tokens = []
        for key in ("id", "status", "progress"):
            value = update.get(key)
            if value is not None:
                tokens.append(value)
        output = ": ".join(tokens)

        log.info(output)

    log.success(f"pulled {repository}:{tag}")
