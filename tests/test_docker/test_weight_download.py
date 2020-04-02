import os

import pytest

from armory import paths
from armory.data.utils import download_file_from_s3


@pytest.mark.usefixtures("ensure_armory_dirs")
def test_download():
    saved_model_dir = paths.docker().saved_model_dir

    weights_file = "resnet50_weights_tf_dim_ordering_tf_kernels.h5"

    filepath = os.path.join(saved_model_dir, weights_file)

    if os.path.isfile(filepath):
        os.remove(filepath)

    download_file_from_s3(
        "armory-public-data",
        f"model-weights/{weights_file}",
        f"{saved_model_dir}/{weights_file}",
    )
    assert os.path.isfile(os.path.join(saved_model_dir, weights_file))
