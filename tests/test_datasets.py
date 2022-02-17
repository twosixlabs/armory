import torch
import numpy as np
import pytest
from armory.data import datasets

# from armory import paths
from armory import paths
import logging
import tensorflow as tf

log = logging.getLogger(__name__)

paths.set_mode("host")
DATASET_DIR = paths.runtime_paths().dataset_dir


def get_generator(name, batch_size, num_epochs, split, framework, dataset_dir):
    # Instance types based on framework
    instance_types = {
        "pytorch": torch.utils.data.DataLoader,
        "tf": (tf.compat.v2.data.Dataset, tf.compat.v1.data.Dataset),
    }

    ds = getattr(datasets, name)
    dataset = ds(
        split=split,
        batch_size=batch_size,
        epochs=num_epochs,
        dataset_dir=dataset_dir,
        framework=framework,
        shuffle_files=False,
        preprocessing_fn=None,
        fit_preprocessing_fn=None,
    )
    assert isinstance(dataset, instance_types[framework])
    return dataset


@pytest.mark.parametrize(
    "name, batch_size, num_epochs, split, framework, xtype, xshape, ytype, yshape",
    [
        # ("cifar10", 16, 1, "train", "pytorch",  torch.uint8, (16, 32, 32, 3), torch.int64, (16,)),
        # ("mnist", 16, 1, "train", "pytorch", torch.uint8, (16, 28, 28, 1), torch.int64, (16,)),
        # ("resisc45", 16, 1, "train", "pytorch", torch.uint8, (16, 256, 256, 3), torch.int64, (16,)),
        ("mnist", 16, 1, "train", "tf", tf.uint8, (16, 28, 28, 1), tf.int64, (16,)),
    ],
)
def test_generator_construction(
    name, batch_size, num_epochs, split, framework, xtype, xshape, ytype, yshape
):
    dataset = get_generator(
        name, batch_size, num_epochs, split, framework, dataset_dir=DATASET_DIR
    )
    images, labels = next(iter(dataset))

    assert images.dtype == xtype
    assert images.shape == xshape
    assert labels.dtype == ytype
    assert labels.shape == yshape


@pytest.mark.parametrize(
    "name, batch_size, num_epochs, split, framework, xtype, xshape, ytype, yshape",
    [
        (
            "mnist",
            10,
            2,
            "test",
            "pytorch",
            torch.uint8,
            (16, 28, 28, 1),
            torch.int64,
            (16,),
        ),
    ],
)
def test_generator_epoch_creation(
    name, batch_size, num_epochs, split, framework, xtype, xshape, ytype, yshape
):
    dataset = get_generator(
        name, batch_size, num_epochs, split, framework, dataset_dir=DATASET_DIR
    )
    cnt = 0

    for images, labels in dataset:
        if cnt % 100 == 0:
            print(cnt, labels)
        if cnt == 0:
            first_batch = labels
        if cnt == 1000:
            second_batch = labels
        cnt += 1

    print(first_batch)
    print("second\n", second_batch)
    assert cnt == 2000
    # TODO Understand why `not` is there...feels like it should be equal
    # assert not torch.all(torch.eq(first_batch, second_batch))


@pytest.mark.parametrize(
    "name, batch_size, framework1, framework2", [("mnist", 10, "tf", "pytorch")]
)
def test_framework_equality(name, batch_size, framework1, framework2):
    ds1 = get_generator(
        name, batch_size, 1, "test", framework1, dataset_dir=DATASET_DIR
    )
    ds2 = get_generator(
        name, batch_size, 1, "test", framework2, dataset_dir=DATASET_DIR
    )

    for ex1, ex2 in zip(ds1, ds2):
        img1, l1 = ex1[0].numpy(), ex1[1].numpy()
        img2, l2 = ex2[0].numpy(), ex2[1].numpy()

        assert np.amax(np.abs(img1 - img2)) == 0
        assert np.amax(np.abs(l1 - l2)) == 0
