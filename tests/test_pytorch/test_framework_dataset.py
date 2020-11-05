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
        split="train",
        epochs=1,
        batch_size=batch_size,
        dataset_dir=DATASET_DIR,
        framework="pytorch",
        preprocessing_fn=None,
        fit_preprocessing_fn=None,
    )

    assert isinstance(dataset, torch.utils.data.DataLoader)
    images, labels = next(iter(dataset))
    assert labels.dtype == torch.int64
    assert labels.shape == (batch_size,)

    assert images.dtype == torch.uint8
    assert images.shape == (batch_size, 32, 32, 3)


def test_pytorch_generator_mnist():
    batch_size = 16
    dataset = datasets.mnist(
        split="train",
        epochs=1,
        batch_size=batch_size,
        dataset_dir=DATASET_DIR,
        framework="pytorch",
        preprocessing_fn=None,
        fit_preprocessing_fn=None,
    )

    assert isinstance(dataset, torch.utils.data.DataLoader)
    images, labels = next(iter(dataset))
    assert labels.dtype == torch.int64
    assert labels.shape == (batch_size,)

    assert images.dtype == torch.uint8
    assert images.shape == (batch_size, 28, 28, 1)


def test_pytorch_generator_resisc():
    batch_size = 16
    dataset = datasets.resisc45(
        split="train",
        epochs=1,
        batch_size=batch_size,
        dataset_dir=DATASET_DIR,
        framework="pytorch",
        preprocessing_fn=None,
        fit_preprocessing_fn=None,
    )

    assert isinstance(dataset, torch.utils.data.DataLoader)
    images, labels = next(iter(dataset))
    assert labels.dtype == torch.int64
    assert labels.shape == (batch_size,)

    assert images.dtype == torch.uint8
    assert images.shape == (batch_size, 256, 256, 3)


def test_pytorch_generator_epochs():
    batch_size = 10
    dataset = datasets.mnist(
        split="test",
        epochs=2,
        batch_size=batch_size,
        dataset_dir=DATASET_DIR,
        framework="pytorch",
        preprocessing_fn=None,
        fit_preprocessing_fn=None,
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
        split="test",
        batch_size=batch_size,
        dataset_dir=DATASET_DIR,
        framework="tf",
        shuffle_files=False,
        preprocessing_fn=None,
        fit_preprocessing_fn=None,
    )

    ds_pytorch = iter(
        datasets.mnist(
            split="test",
            batch_size=batch_size,
            dataset_dir=DATASET_DIR,
            framework="pytorch",
            shuffle_files=False,
            preprocessing_fn=None,
            fit_preprocessing_fn=None,
        )
    )

    for ex_tf, ex_pytorch in zip(ds_tf, ds_pytorch):

        img_tf = ex_tf[0].numpy()
        label_tf = ex_tf[1].numpy()
        img_pytorch = ex_pytorch[0].numpy()
        label_pytorch = ex_pytorch[1].numpy()

        assert np.amax(np.abs(img_tf - img_pytorch)) == 0
        assert np.amax(np.abs(label_tf - label_pytorch)) == 0
