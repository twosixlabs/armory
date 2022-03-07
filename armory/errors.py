import contextlib
import logging

logger = logging.getLogger(__name__)


class ExternalRepoImport(contextlib.AbstractContextManager):
    def __enter__(self, repo="", experiment=""):
        url = f"https://github.com/{repo}"
        name = repo.split("/")[-1].split("@")[0]

        self.error_message = "\n".join(
            [
                f"{name} is an external repo.",
                f"Please download from {url} and place on local environment PYTHONPATH",
                "    OR place in experimental config `external_github_repo` field.",
                f"    See scenario_configs/{experiment} for an example.",
            ]
        )

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if isinstance(exc_type, ImportError):
            logger.exception()
            logger.error(self.error_message)
            return False
        return True
