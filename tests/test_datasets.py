import torch
import numpy as np
import pytest
import logging
import tensorflow as tf

log = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "name, batch_size, num_epochs, split, framework, xtype, xshape, ytype, yshape",
    [
        (
            "cifar10",
            16,
            1,
            "train",
            "pytorch",
            torch.uint8,
            (16, 32, 32, 3),
            torch.int64,
            (16,),
        ),
        (
            "mnist",
            16,
            1,
            "train",
            "pytorch",
            torch.uint8,
            (16, 28, 28, 1),
            torch.int64,
            (16,),
        ),
        (
            "resisc45",
            16,
            1,
            "train",
            "pytorch",
            torch.uint8,
            (16, 256, 256, 3),
            torch.int64,
            (16,),
        ),
        ("mnist", 16, 1, "train", "tf", tf.uint8, (16, 28, 28, 1), tf.int64, (16,)),
        ("mnist", 16, 1, "train", "numpy", tf.uint8, (16, 28, 28, 1), tf.int64, (16,)),
    ],
)
def test_generator_construction(
    name,
    batch_size,
    num_epochs,
    split,
    framework,
    xtype,
    xshape,
    ytype,
    yshape,
    dataset_generator,
    armory_dataset_dir,
):
    dataset = dataset_generator(
        name, batch_size, num_epochs, split, framework, dataset_dir=armory_dataset_dir
    )
    images, labels = next(iter(dataset))

    assert images.dtype == xtype
    assert images.shape == xshape
    assert labels.dtype == ytype
    assert labels.shape == yshape


@pytest.mark.parametrize(
    "name, batch_size, num_epochs, split, framework, shuffle, exp_cnt",
    [("mnist", 10, 2, "test", "pytorch", True, 2000,),],
)
def test_generator_epoch_creation(
    name,
    batch_size,
    num_epochs,
    split,
    framework,
    shuffle,
    exp_cnt,
    dataset_generator,
    armory_dataset_dir,
):
    dataset = dataset_generator(
        name=name,
        batch_size=batch_size,
        num_epochs=num_epochs,
        split=split,
        framework=framework,
        dataset_dir=armory_dataset_dir,
        shuffle_files=shuffle,
    )
    cnt = 0

    for images, labels in dataset:
        if cnt == 0:
            first_batch = labels
        if cnt == 1000:
            second_batch = labels
        cnt += 1

    assert cnt == exp_cnt

    if shuffle:
        assert not torch.all(torch.eq(first_batch, second_batch))
    else:
        assert torch.all(torch.eq(first_batch, second_batch))


@pytest.mark.parametrize(
    "name, batch_size, framework1, framework2", [("mnist", 10, "tf", "pytorch")]
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
