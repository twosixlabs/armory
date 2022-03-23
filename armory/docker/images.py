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
