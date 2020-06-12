import base64
import os
import pathlib
import shutil

from armory import paths
from armory.utils.external_repo import download_and_extract_repos

HOST_PATHS = paths.HostPaths()


def set_github_token():
    """
    Sets a public read-only token to authethenticate our github ci calls. This is only
    done because all VMs for GitHub CI share common IP addresses and thus we can get
    intermittent rate limits should we not authenticate.

    GitHub will revoke the token if it's found in plain-text on a repo. However, this
    token does not need to be hidden.
    """
    b64_key = b"Njc5MjhkMDA0N2Q5ZTBkNTc4MWNmODgxOGE5ZTVlY2JiOWIzMDg2NQ=="
    public_token = base64.b64decode(b64_key).decode()
    os.environ["ARMORY_GITHUB_TOKEN"] = public_token


def test_download():
    set_github_token()
    test_external_repo_dir = pathlib.Path(
        HOST_PATHS.tmp_dir, "test-external-repo-subdir"
    )
    repo = "twosixlabs/armory-example"
    repo_name = repo.split("/")[-1]

    download_and_extract_repos(repo, external_repo_dir=test_external_repo_dir)
    basedir = test_external_repo_dir / repo_name

    assert os.path.exists(basedir)
    assert os.path.isfile(basedir / "README.md")
    shutil.rmtree(basedir)
    os.remove(test_external_repo_dir / (repo_name + ".tar.gz"))
    os.rmdir(test_external_repo_dir)


def test_download_multiple():
    set_github_token()
    test_external_repo_dir = pathlib.Path(
        HOST_PATHS.tmp_dir, "test-external-repo-subdir"
    )
    repos = ["twosixlabs/armory-example", "twosixlabs/armory"]
    repo_names = [x.split("/")[-1] for x in repos]

    download_and_extract_repos(repos, external_repo_dir=test_external_repo_dir)

    for repo_name in repo_names:
        basedir = test_external_repo_dir / repo_name
        assert os.path.exists(basedir)
        assert os.path.isfile(basedir / "README.md")
        shutil.rmtree(basedir)
        os.remove(test_external_repo_dir / (repo_name + ".tar.gz"))
    os.rmdir(test_external_repo_dir)


def test_download_branch():
    set_github_token()
    test_external_repo_dir = pathlib.Path(
        HOST_PATHS.tmp_dir, "test-external-repo-subdir"
    )
    repo = "twosixlabs/armory-example@master"
    org_repo_name = repo.split("@")[0]
    repo_name = org_repo_name.split("/")[-1]

    download_and_extract_repos(repo, external_repo_dir=test_external_repo_dir)
    basedir = test_external_repo_dir / repo_name

    assert os.path.exists(basedir)
    assert os.path.isfile(basedir / "README.md")
    shutil.rmtree(basedir)
    os.remove(test_external_repo_dir / (repo_name + ".tar.gz"))
    os.rmdir(test_external_repo_dir)
