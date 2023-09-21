import os

from armory import __file__ as armory_install_path
from armory.logs import log


def log_current_branch(command_args, prog, description):
    """Log the current git branch of armory. Works independent of the current working directory."""
    try:
        from armory.__about__ import __version__

        log.info(f"Armory version: {__version__}")
    except ModuleNotFoundError:
        log.info("Unable to extract armory version from __about__.py")
    log.info(f"Armory install path: {os.path.dirname(armory_install_path)}")
    try:
        import git

        repo = git.Repo(
            os.path.dirname(os.path.realpath(__file__)), search_parent_directories=True
        )
        log.info(f"Git branch: {repo.active_branch}")
        log.info(
            f"Git commit ({repo.is_dirty() and 'dirty' or 'clean'}): {repo.commit().hexsha}"
        )
    except ImportError:
        log.info(
            "Unable to import gitpython, cannot determine git branch. Please install GitPython."
        )
    except git.exc.InvalidGitRepositoryError:
        log.info("Unable to find .git directory, cannot determine git branch")
    except git.exc.GitCommandError:
        log.info("Unable to determine git branch")
    except Exception as e:
        log.error(f"Unexpected error: {e}")
