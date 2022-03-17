"""
Enables programmatic accessing of most recent docker images
"""

import armory
from armory.logs import log

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
