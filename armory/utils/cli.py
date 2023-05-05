from armory.logs import log


def log_current_branch():
    """Log the current git branch"""
    try:
        import git
    except ImportError:
        log.warning("Unable to import git module")
        return
    try:
        repo = git.Repo(search_parent_directories=True)
        branch_name = repo.active_branch.name
        log.info(f"Git branch: {branch_name}")
    except git.exc.InvalidGitRepositoryError:
        log.warning("Unable to find git repository")
    except git.exc.GitCommandError:
        log.warning("Unable to find git branch")
