import pytest
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import tools.dataset_builder.build as build
import tools.dataset_builder.utils as utils


@pytest.mark.parametrize("dataset_name", ["mnist", "cifar10"])
def test_tfds_build(dataset_name, tmp_path):
    # Making a temp dir to contain fake datasets
    d = tmp_path / "datasets"
    d.mkdir()

    config = utils.SUPPORTED_DATASETS[dataset_name]

    build.build_tfds_dataset(
        dataset_name=dataset_name, local_path=d, feature_dict=config["feature_dict"]
    )
    assert dataset_name in os.listdir(d)
    ds_path = utils.get_dataset_full_path(dataset_name, d)
    print(os.listdir(ds_path))
    utils.validate_dataset_directory_contents(ds_path)


@pytest.mark.parametrize("dataset_name", ["digit"])
def test_local_build(dataset_name, tmp_path):
    # Making a temp dir to contain fake datasets
    d = tmp_path / "datasets"
    d.mkdir()

    config = utils.SUPPORTED_DATASETS[dataset_name]

    build.build_source_dataset(dataset_class_file=config["class_file"], local_path=d)
    assert dataset_name in os.listdir(d)
    ds_path = utils.get_dataset_full_path(dataset_name, d)
    print(os.listdir(ds_path))
    utils.validate_dataset_directory_contents(os.path.join(d, dataset_name))
