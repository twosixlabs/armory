"""
Docker-related utilities
"""

import logging


logger = logging.getLogger(__name__)


def pull_verbose(docker_client, repository, tag=None, log=True):
    """
    Use low-level docker-py API to show status while pulling docker containers.
        Attempts to replicate docker command line output

    log - if True, logs the output, if False, prints the output
    """
    for update in docker_client.api.pull(repository, tag=tag, stream=True, decode=True):
        tokens = []
        for key in ("id", "status", "progress"):
            value = update.get(key)
            if value is not None:
                tokens.append(value)
        output = ": ".join(tokens)

        if log:
            logger.info(output)
        else:
            print(output)
