import pytest
import armory.datasets.builder.utils as utils
import os
import random
import string


def create_fake_name():
    return "".join(random.sample(string.ascii_lowercase, 10))


def create_fake_tfds_dataset(dataset_directory, dataset_name, dataset_version):
    dataset = os.path.join(dataset_directory, dataset_name, dataset_version)
    os.makedirs(dataset)
    for name in [
        "features.json",
        "dataset_info.json",
        f"{dataset_name}-test.tfrecord-00000-of-00001",
    ]:
        with open(os.path.join(dataset, name), "w") as f:
            f.write("# JUST A TEST FILE")
    return dataset


@pytest.mark.parametrize(
    "input,capitalize_first,expected_output",
    [
        ("test_one", True, "TestOne"),
        ("test_one", False, "testOne"),
        ("test_one_ab123", True, "TestOneAb123"),
        ("test_one", True, "TestOne"),
        (
            "ucf101_mars_perturbation_and_patch_adversarial112x112",
            True,
            "Ucf101MarsPerturbationAndPatchAdversarial112x112",
        ),
    ],
)
def test_camel_case(input, capitalize_first, expected_output):
    output = utils.camel_case(input, capitalize_first)
    assert output == expected_output


@pytest.mark.parametrize(
    "cls_file, expected_output",
    [
        (
            os.path.join(
                os.path.dirname(__file__),
                "resources",
                "example_local_class_directory",
                "my_dataset.py",
            ),
            {
                "type": "source",
                "class_file": os.path.join(
                    os.path.dirname(__file__),
                    "resources",
                    "example_local_class_directory",
                    "my_dataset.py",
                ),
                "expected_name": "MyDataset",
                "expected_version": "1.0.0",
            },
        ),
        (
            os.path.join(
                os.path.dirname(__file__),
                "../../armory/datasets/builder",
                "build_classes",
                "digit.py",
            ),
            {
                "type": "source",
                "class_file": os.path.join(
                    os.path.dirname(__file__),
                    "../../armory/datasets/builder",
                    "build_classes",
                    "digit.py",
                ),
                "expected_name": "Digit",
                "expected_version": "1.0.8",
            },
        ),
    ],
)
def test_get_local_config(cls_file, expected_output):
    output = utils.get_local_config(cls_file)
    assert output == expected_output


@pytest.mark.parametrize(
    "ds_full_path",
    [
        os.path.join(
            os.path.dirname(__file__),
            "resources",
            "example_built_datasets",
            "my_dataset",
        ),
        os.path.join(
            os.path.dirname(__file__),
            "resources",
            "example_built_datasets",
            "my_dataset",
            "1.0.0",
        ),
    ],
)
def test_validate_dataset_directory_contents(ds_full_path):
    utils.validate_dataset_directory_contents(ds_full_path)


@pytest.mark.parametrize(
    "dataset_name, dataset_directory, validate, expected_output",
    [
        (
            "my_dataset",
            os.path.join(
                os.path.dirname(__file__), "resources", "example_built_datasets"
            ),
            True,
            os.path.join(
                os.path.dirname(__file__),
                "resources",
                "example_built_datasets",
                "my_dataset",
                "1.0.0",
            ),
        )
    ],
)
def test_get_dataset_full_path(
    dataset_name, dataset_directory, validate, expected_output
):
    pth = utils.get_dataset_full_path(dataset_name, dataset_directory, validate)
    assert pth == expected_output


@pytest.mark.parametrize(
    "dataset_name, dataset_version, expected_output",
    [
        ("my_dataset", "1.0.0", "my_dataset_1.0.0.tar.gz"),
        ("so2sat/all", "3.5.4", "so2sat_all_3.5.4.tar.gz"),
    ],
)
def test_archive_name(dataset_name, dataset_version, expected_output):
    name = utils.get_dataset_archive_name(dataset_name, dataset_version)
    assert name == expected_output


def test_resolve_dataset_directories(tmp_path):

    # Making a temp dir to contain fake datasets
    d = tmp_path / "datasets"
    d.mkdir()

    fake_names = [create_fake_name(), create_fake_name()]
    fake_versions = ["1.0.0", "4.0.2"]
    ds_paths = [
        create_fake_tfds_dataset(d, n, v) for n, v in zip(fake_names, fake_versions)
    ]

    dirs = utils.resolve_dataset_directories(datasets=ds_paths)
    dirs2 = utils.resolve_dataset_directories(parents=[d])
    assert set(dirs) == set(dirs2)


def test_load_from_directory():
    example_dataset_directory = os.path.join(
        os.path.dirname(__file__),
        "resources",
        "example_built_datasets",
        "my_dataset",
        "1.0.0",
    )
    ds_info, ds = utils.load_from_directory(example_dataset_directory)
    assert ds_info.name == "my_dataset"
    assert ds_info.full_name == "my_dataset/1.0.0"
    sample = next(iter(ds["train"]))
    assert "image" in sample.keys()
    assert "label" in sample.keys()

    assert sample["image"].shape == (512, 640, 3)
    assert sample["label"].numpy() == 1


def test_load():
    example_dataset_directory = os.path.join(
        os.path.dirname(__file__), "resources", "example_built_datasets"
    )
    ds_info, ds = utils.load("my_dataset", example_dataset_directory)
    assert ds_info.name == "my_dataset"
    assert ds_info.full_name == "my_dataset/1.0.0"
    sample = next(iter(ds["train"]))
    assert "image" in sample.keys()
    assert "label" in sample.keys()

    assert sample["image"].shape == (512, 640, 3)
    assert sample["label"].numpy() == 1
