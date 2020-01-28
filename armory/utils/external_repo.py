"""

"""
import os
import logging
import tarfile
import shutil

import coloredlogs
import requests

coloredlogs.install(level=logging.DEBUG)

logger = logging.getLogger(__name__)


def download_and_extract(config: dict) -> None:
    """
    Downloads and extracts an external repository for use within ARMORY.

    Private repositories require a `GITHUB_TOKEN` environment variable.
    :param config: Dictionary of loaded configuration info
    """
    os.makedirs("external_repos", exist_ok=True)
    headers = {}
    external_repo = config["external_github_repo"]
    repo_name = external_repo.split("/")[-1]

    if "GITHUB_TOKEN" in os.environ:
        headers = {"Authorization": f'token {os.getenv("GITHUB_TOKEN")}'}

    response = requests.get(
        f"https://api.github.com/repos/{external_repo}/tarball/master",
        headers=headers,
        stream=True,
    )

    if response.status_code == 200:
        logging.info(f"Downloading external repo: {external_repo}")

        tar_filename = repo_name + ".tar.gz"
        with open(tar_filename, "wb") as f:
            f.write(response.raw.read())
        tar = tarfile.open(tar_filename, "r:gz")
        dl_directory_name = tar.getnames()[0]
        tar.extractall(path="external_repos")

        # Always overwrite existing repositories to keep them at HEAD
        final_dir_name = f"external_repos/{repo_name}"
        if os.path.isdir(final_dir_name):
            shutil.rmtree(final_dir_name)
        os.rename(f"external_repos/{dl_directory_name}", final_dir_name)
        os.remove(tar_filename)

    else:
        raise ConnectionError(
            "Unable to download repository. If it's private make sure "
            "`GITHUB_TOKEN` environment variable is set"
        )
