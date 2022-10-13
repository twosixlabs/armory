import os
import shutil

import pytest

from armory.data.utils import download_file_from_s3, maybe_download_weights_from_s3
from armory.utils.external_repo import download_and_extract_repos

pytestmark = [pytest.mark.online]


def test_valid_model_weight(tmp_path):

    dir = tmp_path / "saved_model_dir"  # tmp_path is pytest Pathlib fixture
    dir.mkdir()
    print("Saving Model to Tmp Directory: {}".format(dir))
    # saved_model_dir = paths.runtime_paths().saved_model_dir

    weights_file = "resnet50_imagenet_v1.h5"

    filepath = os.path.join(dir, weights_file)

    if os.path.isfile(filepath):
        raise Exception(
            "This should never happen because tmp_path is created at runtime"
        )

    download_file_from_s3(
        "armory-public-data",
        f"model-weights/{weights_file}",
        filepath,
    )
    assert os.path.isfile(filepath)
    shutil.rmtree(str(dir))


def test_invalid_model_weight():
    weights_file = "does_not_exist.h5"
    with pytest.raises(ValueError, match="attempting to load a custom set of weights"):
        maybe_download_weights_from_s3(weights_file)


@pytest.mark.parametrize(
    "name, repo",
    [
        ("single", "twosixlabs/armory-example"),
        ("single branch", "twosixlabs/armory-example@master"),
        ("single in list", ["twosixlabs/armory-example"]),
        ("multiple in list", ["twosixlabs/armory-example", "twosixlabs/armory"]),
    ],
)
def test_valid_external_github_repo(name, repo, tmp_path):
    dir = tmp_path / "test-external-repo-dir"

    download_and_extract_repos(repo, external_repo_dir=dir)
    if isinstance(repo, str):
        repo_list = [repo]
    else:
        repo_list = repo

    for rp in repo_list:

        repo_name = rp.split("@")[0].split("/")[-1]
        basedir = dir / repo_name
        print(basedir)
        print(os.listdir(basedir))
        assert os.path.exists(str(basedir))
        assert os.path.isfile(str(basedir / "README.md"))
        shutil.rmtree(basedir)
    shutil.rmtree(dir)
