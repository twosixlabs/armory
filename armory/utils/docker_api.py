"""
Docker-related utilities
"""

from armory.logs import log, is_progress


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
