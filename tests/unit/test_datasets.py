import torch
import numpy as np
import pytest
import tensorflow as tf


# TODO Right now unit tests only run for `mnist`, `cifar10`, and `resisc10`
#  figure out if this is what we want or if we need more


@pytest.mark.parametrize("name", ["cifar10", "mnist", "resisc10"])
@pytest.mark.parametrize("split", ["test", "train", "validation"])
@pytest.mark.parametrize("framework", ["numpy", "tf", "pytorch"])
def test_dataset_generation(
    name, split, framework, armory_dataset_dir, dataset_generator
):
    if split == "validation" and name in ["cifar10", "mnist"]:
        pytest.skip("Dataset: {} Does not have Split: {}".format(name, split))

    # Expected Output Parameters
    parameters = {
        "cifar10": {
            "sizes": {"train": 50000, "test": 10000},
            "shapes": ((32, 32, 3), ()),
        },
        "mnist": {
            "sizes": {"train": 60000, "test": 10000},
            "shapes": ((28, 28, 1), ()),
        },
        "resisc10": {
            "sizes": {"train": 5000, "test": 1000, "validation": 1000},
            "shapes": ((256, 256, 3), ()),
        },
    }

    batch_size = 16
    dataset = dataset_generator(
        name, batch_size, 1, split, framework, dataset_dir=armory_dataset_dir
    )

    x, y = next(iter(dataset))
    typehead = torch if framework == "pytorch" else tf

    assert x.dtype == typehead.uint8
    assert y.dtype == typehead.int64

    assert x.shape == (batch_size,) + parameters[name]["shapes"][0]
    assert y.shape == (batch_size,) + parameters[name]["shapes"][1]

    if framework == "numpy":
        assert dataset.size == parameters[name]["sizes"][split]
        x, y = dataset.get_batch()
        assert x.shape == (batch_size,) + parameters[name]["shapes"][0]
        assert y.shape == (batch_size,) + parameters[name]["shapes"][1]
        assert isinstance(x, np.ndarray)


@pytest.mark.parametrize(
    "name, batch_size, framework1, framework2",
    [
        ("mnist", 10, "tf", "pytorch"),
        ("cifar10", 10, "tf", "pytorch"),
        ("resisc10", 10, "tf", "pytorch"),
    ],
)
def test_framework_equality(
    name, batch_size, framework1, framework2, dataset_generator, armory_dataset_dir
):
    ds1 = dataset_generator(
        name, batch_size, 1, "test", framework1, dataset_dir=armory_dataset_dir
    )
    ds2 = dataset_generator(
        name, batch_size, 1, "test", framework2, dataset_dir=armory_dataset_dir
    )

    for ex1, ex2 in zip(ds1, ds2):
        img1, l1 = ex1[0].numpy(), ex1[1].numpy()
        img2, l2 = ex2[0].numpy(), ex2[1].numpy()

        assert np.amax(np.abs(img1 - img2)) == 0
        assert np.amax(np.abs(l1 - l2)) == 0
