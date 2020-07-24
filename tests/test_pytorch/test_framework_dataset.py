"""
Test cases for framework specific ARMORY datasets.
"""

import torch
import numpy as np

from armory.data import datasets
from armory import paths

DATASET_DIR = paths.DockerPaths().dataset_dir


def test_pytorch_generator_cifar10():
    batch_size = 16
    dataset = datasets.cifar10(
        split_type="train",
        epochs=1,
        batch_size=batch_size,
        dataset_dir=DATASET_DIR,
        framework="pytorch",
    )

    assert isinstance(dataset, torch.utils.data.DataLoader)
    labels, images = next(iter(dataset))
    assert labels.dtype == torch.int64
    assert labels.shape == (batch_size,)

    assert images.dtype == torch.uint8
    assert images.shape == (batch_size, 32, 32, 3)


def test_pytorch_generator_mnist():
    batch_size = 16
    dataset = datasets.mnist(
        split_type="train",
        epochs=1,
        batch_size=batch_size,
        dataset_dir=DATASET_DIR,
        framework="pytorch",
    )

    assert isinstance(dataset, torch.utils.data.DataLoader)
    labels, images = next(iter(dataset))
    assert labels.dtype == torch.int64
    assert labels.shape == (batch_size,)

    assert images.dtype == torch.uint8
    assert images.shape == (batch_size, 28, 28, 1)


def test_pytorch_generator_resisc():
    batch_size = 16
    dataset = datasets.resisc45(
        split_type="train",
        epochs=1,
        batch_size=batch_size,
        dataset_dir=DATASET_DIR,
        framework="pytorch",
    )

    assert isinstance(dataset, torch.utils.data.DataLoader)
    labels, images = next(iter(dataset))
    assert labels.dtype == torch.int64
    assert labels.shape == (batch_size,)

    assert images.dtype == torch.uint8
    assert images.shape == (batch_size, 256, 256, 3)


def test_pytorch_generator_epochs():
    batch_size = 10
    dataset = datasets.mnist(
        split_type="test",
        epochs=2,
        batch_size=batch_size,
        dataset_dir=DATASET_DIR,
        framework="pytorch",
    )

    cnt = 0
    for images, labels in dataset:
        if cnt == 0:
            first_batch = labels

        if cnt == 1000:
            second_batch = labels
        cnt += 1

    assert cnt == 2000
    assert not torch.all(torch.eq(first_batch, second_batch))


def test_tf_pytorch_equality():

    batch_size = 10
    ds_tf = datasets.mnist(
        split_type="test",
        batch_size=batch_size,
        dataset_dir=DATASET_DIR,
        framework="tf",
        shuffle_files=False,
    )

    ds_pytorch = iter(
        datasets.mnist(
            split_type="test",
            batch_size=batch_size,
            dataset_dir=DATASET_DIR,
            framework="pytorch",
            shuffle_files=False,
        )
    )

    for img_tf, label_tf in ds_tf:
        label_pytorch, img_pytorch = next(ds_pytorch)

        img_tf = img_tf.numpy()
        label_tf = label_tf.numpy()
        img_pytorch = img_pytorch.numpy()
        label_pytorch = label_pytorch.numpy()

        assert np.amax(np.abs(img_tf - img_pytorch)) == 0
        assert np.amax(np.abs(label_tf - label_pytorch)) == 0
