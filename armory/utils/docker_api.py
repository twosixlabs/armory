"""
Docker-related utilities
"""

from armory.logs import log as logger


def pull_verbose(docker_client, repository, tag=None):
    """
    Use low-level docker-py API to show status while pulling docker containers.
        Attempts to replicate docker command line output
    """
    for update in docker_client.api.pull(repository, tag=tag, stream=True, decode=True):
        tokens = []
        for key in ("id", "status", "progress"):
            value = update.get(key)
            if value is not None:
                tokens.append(value)
        output = ": ".join(tokens)

        logger.info(output)
