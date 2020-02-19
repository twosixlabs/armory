"""
Enables programmatic accessing of most recent docker images
"""

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
    if tokens[0] in REPOSITORIES:
        return True
    return False
