"""
Utils to pull external repos for evaluation
"""
import os
import logging
import tarfile
import shutil
import sys
from typing import Union, List

import requests

from armory import paths
from armory.configuration import get_verify_ssl


logger = logging.getLogger(__name__)


def download_and_extract_repos(
    external_repos: Union[str, List[str]], external_repo_dir: str = None
):
    """
    Conditionally download and extract a single repo or a list of repos
    """
    if isinstance(external_repos, str):
        download_and_extract_repo(external_repos, external_repo_dir)
    elif isinstance(external_repos, list):
        for repo in external_repos:
            download_and_extract_repo(repo, external_repo_dir)
    else:
        raise ValueError(
            "`external_repos` must be of a repo string or list of repo strings"
        )


def download_and_extract_repo(
    external_repo_name: str, external_repo_dir: str = None
) -> None:
    """
    Downloads and extracts an external repository for use within ARMORY. The external
    repositories project root will be added to the sys path.

    Private repositories require an `ARMORY_GITHUB_TOKEN` environment variable.
    :param external_repo_name: String name of "organization/repo-name" or "organization/repo-name@branch"
    """
    verify_ssl = get_verify_ssl()

    if external_repo_dir is None:
        external_repo_dir = paths.HostPaths().external_repo_dir

    os.makedirs(external_repo_dir, exist_ok=True)
    headers = {}

    if "@" in external_repo_name:
        org_repo_name, branch = external_repo_name.split("@")
    else:
        org_repo_name = external_repo_name
        branch = "master"
    repo_name = org_repo_name.split("/")[-1]

    if "ARMORY_GITHUB_TOKEN" in os.environ and os.getenv("ARMORY_GITHUB_TOKEN") != "":
        headers = {"Authorization": f'token {os.getenv("ARMORY_GITHUB_TOKEN")}'}

    response = requests.get(
        f"https://api.github.com/repos/{org_repo_name}/tarball/{branch}",
        headers=headers,
        stream=True,
        verify=verify_ssl,
    )

    if response.status_code == 200:
        logging.info(f"Downloading external repo: {external_repo_name}")

        tar_filename = os.path.join(external_repo_dir, repo_name + ".tar.gz")
        with open(tar_filename, "wb") as f:
            f.write(response.raw.read())
        tar = tarfile.open(tar_filename, "r:gz")
        dl_directory_name = tar.getnames()[0]
        tar.extractall(path=external_repo_dir)

        # Always overwrite existing repositories to keep them at HEAD
        final_dir_name = os.path.join(external_repo_dir, repo_name)
        if os.path.isdir(final_dir_name):
            shutil.rmtree(final_dir_name)
        os.rename(
            os.path.join(external_repo_dir, dl_directory_name), final_dir_name,
        )
        sys.path.insert(0, external_repo_dir)
        sys.path.insert(0, final_dir_name)

    else:
        raise ConnectionError(
            "Unable to download repository. If it's private make sure "
            "`ARMORY_GITHUB_TOKEN` environment variable is set\n"
            f"status_code is {response.status_code}\n"
            f"full response is {response.text}"
        )


def add_local_repo(local_repo_name: str) -> None:
    local_repo_dir = os.path.join(paths.runtime_paths().tmp_dir, "local")
    if not os.path.isdir(local_repo_dir):
        raise ValueError(f"{local_repo_dir} is not a valid directory path")
    if local_repo_dir not in sys.path:
        sys.path.insert(0, local_repo_dir)
        logger.info(f"Added {local_repo_dir} to PYTHONPATH")

    local_repo_path = os.path.join(local_repo_dir, local_repo_name)
    if not os.path.isdir(local_repo_path):
        raise ValueError(f"{local_repo_path} is not a valid directory path")
    if local_repo_path not in sys.path:
        sys.path.insert(0, local_repo_path)
        logger.info(f"Added {local_repo_path} to PYTHONPATH")

    local_repo_parent_path = os.path.dirname(local_repo_path)
    if local_repo_parent_path not in sys.path:
        sys.path.insert(0, local_repo_parent_path)
        logger.info(f"Added {local_repo_parent_path} to PYTHONPATH")


def add_pythonpath(subpath: str, external_repo_dir: str = None) -> None:
    if external_repo_dir is None:
        external_repo_dir = paths.HostPaths().external_repo_dir

    subpath = subpath.rstrip("/")
    # Add primary path and root path (for module import)
    repo_base = os.path.join(external_repo_dir, subpath)
    if not os.path.isdir(repo_base):
        raise ValueError(f"{repo_base} is not a valid directory path")
    if repo_base not in sys.path:
        sys.path.insert(0, repo_base)

    top_level = os.path.dirname(repo_base)
    if not os.path.isdir(top_level):
        raise ValueError(f"{top_level} is not a valid directory path")
    if top_level not in sys.path:
        sys.path.insert(0, top_level)
