import os
import shutil
import pytest
from armory.data.utils import download_file_from_s3, maybe_download_weights_from_s3

# TODO:  Notes - This originated from tests/test_docker/test_weight_download.py
#  but it seemed strange to keep it there so promoted it up and added more tests
#  for downloading


def test_download(tmp_path):

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
        os.remove(filepath)

    download_file_from_s3(
        "armory-public-data", f"model-weights/{weights_file}", filepath,
    )
    assert os.path.isfile(filepath)
    shutil.rmtree(str(dir))


def test_invalid_weights():
    weights_file = "does_not_exist.h5"
    with pytest.raises(ValueError, match="attempting to load a custom set of weights"):
        maybe_download_weights_from_s3(weights_file)
