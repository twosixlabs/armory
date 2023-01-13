"""
Enables programmatic accessing of most recent docker images
"""

import requests

import armory
from armory.logs import is_progress, log
from armory.utils import version
import docker
import docker.errors

log.trace(f"armory.__version__: {armory.__version__}")

TAG = version.to_docker_tag(armory.__version__)
ARMORY_IMAGE_NAME = f"twosixarmory/armory:{TAG}"
DEEPSPEECH_IMAGE_NAME = f"twosixarmory/pytorch-deepspeech:{TAG}"

IMAGE_MAP = {
    "armory": ARMORY_IMAGE_NAME,
    "tf2": ARMORY_IMAGE_NAME,
    "pytorch": ARMORY_IMAGE_NAME,
    "carla-mot": ARMORY_IMAGE_NAME,
    "pytorch-deepspeech": DEEPSPEECH_IMAGE_NAME,
}


def split_name(name: str):
    """
    Return the components of user/repo:tag as (user, repo, tag)
        Return the empty string "" for any that do not exist
    """
    if ":" in name:
        user_repo, tag = name.split(":")
    else:
        user_repo, tag = name, ""

    if "/" in user_repo:
        i = user_repo.rindex("/")
        user, repo = user_repo[:i], user_repo[i + 1 :]
    else:
        user, repo = "", user_repo

    return user, repo, tag


def join_name(user: str, repo: str, tag: str):
    """
    Inverse of split_image_name. The following should be identity operations:
        image_name = join_name(*split_name(image_name))
        user, repo, tag = split_name(join_name(user, repo, tag))
    """
    if user:
        user = user + "/"
    if tag:
        tag = ":" + tag
    return f"{user}{repo}{tag}"


def is_armory(image_name: str):
    """
    Return whether image_name refers to an armory docker image
    """
    user, repo, _ = split_name(image_name)
    if user and user != "twosixarmory":
        return False
    if repo == "tf1":
        raise ValueError("tf1 docker image is deprecated. Use Armory version < 0.15.0")
    return repo in IMAGE_MAP


def get_armory_name(image_name: str):
    """
    Takes a docker image name, returns False if not a armory image
        If an armory image, it will return `twosixarmory/<framework>:tag`
    """
    if not is_armory(image_name):
        raise ValueError(f"Not an armory image name: {image_name}")
    user, repo, tag = split_name(image_name)
    user = "twosixarmory"
    if tag:  # tag is explicitly defined, use it
        tag = version.to_docker_tag(tag)
        return join_name(user, repo, tag)
    return IMAGE_MAP[repo]


def last_armory_release(image_name: str):
    """
    Return the image_name corresponding to the last armory major.minor.patch release
        If the current image_name is a release, return the current
    """
    user, repo, tag = split_name(image_name)
    if not user or not tag:
        raise ValueError("Must be a full user/repo:tag docker image_name")
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
        return join_name(user, repo, release_tag)
    elif len(tokens) in (5, 6):
        major, minor, patch, post, ghash = tokens[:5]
        if len(tokens) == 6:
            date = tokens[5]
            if not date.startswith("d20"):
                raise ValueError(f"Tag {tag} date must start with 'd20'")
        if not post.startswith("post"):
            raise ValueError(f"Tag {tag} post must start with 'post'")
        if not ghash.startswith("g"):
            raise ValueError(f"Tag {tag} git hash must start with 'g'")
        release_tag = ".".join([major, minor, patch])
        return join_name(user, repo, release_tag)
    else:
        raise ValueError(f"Tag {tag} must be in major.minor.patch[.SCM version format]")


def is_image_local(docker_client, image_name):
    """
    Return True if image_name is found, else False
    """
    log.trace(f"asking local docker for image {image_name}")
    try:
        docker_client.images.get(image_name)
        log.success(f"found docker image {image_name}")
        return True
    except docker.errors.ImageNotFound:
        log.trace(f"image {image_name} not found")
        return False


def ensure_image_present(image_name: str) -> str:
    """
    If image_name is available, return it. Otherwise, pull it from dockerhub.
    """
    log.trace(f"ensure_image_present {image_name}")
    docker_client = docker.from_env()

    if not is_armory(image_name):
        if is_image_local(docker_client, image_name):
            return image_name

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
        if is_image_local(docker_client, canon_image_name):
            return canon_image_name

        user, repo, tag = split_name(canon_image_name)
        tokens = tag.split(".")
        if len(tokens) == 6:
            tokens = tokens[:5]
            tag = ".".join(tokens)
            clean_canon_image_name = join_name(user, repo, tag)
            log.info(
                f"Current workdir is dirty. Reverting to non-dirty image {clean_canon_image_name}"
            )
            if is_image_local(docker_client, clean_canon_image_name):
                return clean_canon_image_name

        log.info(f"reverting to previous release tag image {prev_release}")

    if is_image_local(docker_client, prev_release):
        return prev_release

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
    if tag is None and ":" in repository:
        repository, tag = repository.split(":")
    elif ":" in repository:
        raise ValueError(
            f"cannot set tag kwarg and have tag in repository arg {repository}"
        )
    elif tag is None:
        log.info("empty tag is set to latest by API")
        tag = "latest"

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
