"""
Utils to pull external repos for evaluation
"""
import contextlib
import os
import shutil
import sys
import tarfile
from typing import List, Union

import requests

from armory import paths
from armory.configuration import get_verify_ssl
from armory.logs import log


class ExternalRepoImport(contextlib.AbstractContextManager):
    def __init__(self, repo="", experiment=""):
        super().__init__()
        url = f"https://github.com/{repo}"
        name = repo.split("/")[-1].split("@")[0]
        error_message_lines = [
            f"{name} is an external repo.",
            f"Please download from {url} and place on local environment PYTHONPATH",
            "    OR place in experimental config `external_github_repo` field.",
        ]
        if experiment:
            error_message_lines.append(
                f"    See scenario_configs/{experiment} for an example."
            )
        self.error_message = "\n".join(error_message_lines)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None and issubclass(exc_type, ImportError):
            log.error(self.error_message)
            return False
        return True


class ExternalPipInstalledImport(contextlib.AbstractContextManager):
    def __init__(self, package="", dockerimage=""):
        super().__init__()
        error_message_lines = [
            f"{package} is an external dependency.",
            f"Please 'pip install {package}' in local environment",
        ]
        if dockerimage:
            error_message_lines.append(f"    OR use docker image {dockerimage}")
        self.error_message = "\n".join(error_message_lines)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None and issubclass(exc_type, ImportError):
            log.error(self.error_message)
            return False
        return True


def add_path(path, include_parent=False, index=1):
    """
    Add path to the PYTHONPATH, inserted after current working directory

    If include_parent, parent is also included and is inserted first
    """
    path = path.rstrip("/")
    if include_parent:
        parent_path = os.path.dirname(path)
        add_path(parent_path, include_parent=False, index=index)

    if not os.path.isdir(path):
        raise ValueError(f"{path} is not a valid directory path")
    if path not in sys.path:
        sys.path.insert(index, path)


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
        external_repo_dir = paths.runtime_paths().external_repo_dir

    os.makedirs(external_repo_dir, exist_ok=True)
    headers = {}

    if "@" in external_repo_name:
        org_repo_name, branch = external_repo_name.split("@")
    else:
        org_repo_name = external_repo_name
        branch = ""
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
        log.info(f"Downloading external repo: {external_repo_name}")

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
            os.path.join(external_repo_dir, dl_directory_name),
            final_dir_name,
        )
        add_path(final_dir_name, include_parent=True)

    else:
        raise ConnectionError(
            "Unable to download repository. If it's private make sure "
            "`ARMORY_GITHUB_TOKEN` environment variable is set\n"
            f"status_code is {response.status_code}\n"
            f"full response is {response.text}"
        )


def add_local_repo(local_repo_name: str) -> None:
    local_repo_dir = paths.runtime_paths().local_git_dir
    path = os.path.join(local_repo_dir, local_repo_name)
    add_path(path, include_parent=True)


def add_pythonpath(subpath: str, external_repo_dir: str = None) -> None:
    if external_repo_dir is None:
        external_repo_dir = paths.runtime_paths().external_repo_dir

    path = os.path.join(external_repo_dir, subpath)
    add_path(path, include_parent=True)
