import os

import pytest

from armory import paths
from armory.data.utils import download_file_from_s3, maybe_download_weights_from_s3


@pytest.mark.usefixtures("ensure_armory_dirs")
def test_download():
    saved_model_dir = paths.DockerPaths().saved_model_dir

    weights_file = "resnet50_imagenet_v1.h5"

    filepath = os.path.join(saved_model_dir, weights_file)

    if os.path.isfile(filepath):
        os.remove(filepath)

    download_file_from_s3(
        "armory-public-data",
        f"model-weights/{weights_file}",
        f"{saved_model_dir}/{weights_file}",
    )
    assert os.path.isfile(os.path.join(saved_model_dir, weights_file))


@pytest.mark.usefixtures("ensure_armory_dirs")
def test_invalid_weights():
    weights_file = "does_not_exist.h5"
    with pytest.raises(ValueError, match="attempting to load a custom set of weights"):
        maybe_download_weights_from_s3(weights_file)
