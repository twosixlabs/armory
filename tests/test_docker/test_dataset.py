"""
Test cases for ARMORY datasets.
"""

import os

import pytest

from armory.data import datasets
from armory import paths

DATASET_DIR = paths.docker().dataset_dir


def test_mnist():
    batch_size = 600
    for split, size in [("train", 60000), ("test", 10000)]:
        dataset = datasets.mnist(
            split_type=split, epochs=1, batch_size=batch_size, dataset_dir=DATASET_DIR,
        )
        assert dataset.size == size
        assert dataset.batch_size == batch_size
        assert dataset.batches_per_epoch == (
            size // batch_size + bool(size % batch_size)
        )

        x, y = dataset.get_batch()
        assert x.shape == (batch_size, 28, 28, 1)
        assert y.shape == (batch_size,)


def test_cifar():
    batch_size = 500
    for split, size in [("train", 50000), ("test", 10000)]:
        dataset = datasets.cifar10(
            split_type=split, epochs=1, batch_size=batch_size, dataset_dir=DATASET_DIR,
        )
        assert dataset.size == size
        assert dataset.batch_size == batch_size
        assert dataset.batches_per_epoch == (
            size // batch_size + bool(size % batch_size)
        )

        x, y = dataset.get_batch()
        assert x.shape == (batch_size, 32, 32, 3)
        assert y.shape == (batch_size,)


def test_digit():
    epochs = 1
    batch_size = 1
    num_users = 3
    min_length = 1148
    max_length = 18262
    for split, size in [
        ("train", 45 * num_users * 10),
        ("test", 5 * num_users * 10),
    ]:
        dataset = datasets.digit(
            split_type=split,
            epochs=epochs,
            batch_size=batch_size,
            dataset_dir=DATASET_DIR,
        )
        assert dataset.size == size
        assert dataset.batch_size == batch_size

        x, y = dataset.get_batch()
        assert x.shape[0] == batch_size
        assert x.ndim == 2
        assert min_length <= x.shape[1] <= max_length
        assert y.shape == (batch_size,)


def test_imagenet_adv():
    batch_size = 100
    total_size = 1000
    test_dataset = datasets.imagenet_adversarial(
        dataset_dir=DATASET_DIR, split_type="clean", batch_size=batch_size, epochs=1,
    )
    assert test_dataset.size == total_size
    assert test_dataset.batch_size == batch_size
    assert test_dataset.batches_per_epoch == (
        total_size // batch_size + bool(total_size % batch_size)
    )

    x, y = test_dataset.get_batch()
    assert x.shape == (batch_size, 224, 224, 3)
    assert y.shape == (batch_size,)


def test_german_traffic_sign():
    for split, size in [("train", 39209), ("test", 12630)]:
        batch_size = 1
        epochs = 1
        dataset = datasets.german_traffic_sign(
            split_type=split,
            epochs=epochs,
            batch_size=batch_size,
            dataset_dir=DATASET_DIR,
        )
        assert dataset.size == size

        x, y = dataset.get_batch()
        # sign image shape is variable so we don't compare 2nd dim
        assert x.shape[:1] + x.shape[3:] == (batch_size, 3)
        assert y.shape == (batch_size,)


def test_imagenette():
    if not os.path.isdir(os.path.join(DATASET_DIR, "imagenette", "full-size", "0.1.0")):
        pytest.skip("imagenette dataset not locally available.")

    for split, size in [("train", 12894), ("validation", 500)]:
        batch_size = 1
        epochs = 1
        dataset = datasets.imagenette(
            split_type=split,
            epochs=epochs,
            batch_size=batch_size,
            dataset_dir=DATASET_DIR,
        )
        assert dataset.size == size

        x, y = dataset.get_batch()
        # image dimensions are variable so we don't compare 2nd dim or 3rd dim
        assert x.shape[:1] + x.shape[3:] == (batch_size, 3)
        assert y.shape == (batch_size,)


def test_ucf101():
    if not os.path.isdir(os.path.join(DATASET_DIR, "ucf101", "ucf101_1", "2.0.0")):
        pytest.skip("ucf101 dataset not locally available.")

    for split, size in [("train", 9537), ("test", 3783)]:
        batch_size = 1
        epochs = 1
        dataset = datasets.ucf101(
            split_type=split,
            epochs=epochs,
            batch_size=batch_size,
            dataset_dir=DATASET_DIR,
        )
        assert dataset.size == size

        x, y = dataset.get_batch()
        # video length is variable so we don't compare 2nd dim
        assert x.shape[:1] + x.shape[2:] == (batch_size, 240, 320, 3)
        assert y.shape == (batch_size,)


def test_librispeech():
    if not os.path.exists(os.path.join(DATASET_DIR, "librispeech_dev_clean_split")):
        pytest.skip("Librispeech dataset not downloaded.")

    splits = ("train", "validation", "test")
    sizes = (1371, 692, 640)
    min_dim1s = (23120, 26239, 24080)
    max_dim1s = (519760, 516960, 522320)
    batch_size = 1

    for split, size, min_dim1, max_dim1 in zip(splits, sizes, min_dim1s, max_dim1s):
        dataset = datasets.librispeech_dev_clean(
            split_type=split, epochs=1, batch_size=batch_size, dataset_dir=DATASET_DIR,
        )
        assert dataset.size == size
        assert dataset.batch_size == batch_size
        assert dataset.batches_per_epoch == (
            size // batch_size + bool(size % batch_size)
        )

        x, y = dataset.get_batch()
        assert x.shape[0] == 1
        assert min_dim1 <= x.shape[1] <= max_dim1
        assert y.shape == (batch_size,)


def test_resisc45():
    """
    Skip test if not locally available
    """
    if not os.path.isdir(os.path.join(DATASET_DIR, "resisc45_split", "3.0.0")):
        pytest.skip("resisc45_split dataset not locally available.")

    for split, size in [("train", 22500), ("validation", 4500), ("test", 4500)]:
        batch_size = 16
        epochs = 1
        dataset = datasets.resisc45(
            split_type=split,
            epochs=epochs,
            batch_size=batch_size,
            dataset_dir=DATASET_DIR,
        )
        assert dataset.size == size
        assert dataset.batch_size == batch_size
        assert dataset.batches_per_epoch == (
            size // batch_size + bool(size % batch_size)
        )

        x, y = dataset.get_batch()
        assert x.shape == (batch_size, 256, 256, 3)
        assert y.shape == (batch_size,)


def test_variable_length():
    """
    Test batches with variable length items using digit dataset
    """
    size = 1350
    batch_size = 4
    dataset = datasets.digit(
        split_type="train", epochs=1, batch_size=batch_size, dataset_dir=DATASET_DIR,
    )
    assert dataset.batches_per_epoch == (size // batch_size + bool(size % batch_size))

    x, y = dataset.get_batch()
    assert x.dtype == object
    assert x.shape == (batch_size,)
    for x_i in x:
        assert x_i.ndim == 1
        assert 1148 <= len(x_i) <= 18262
    assert y.shape == (batch_size,)


def test_generator():
    batch_size = 600
    for split, size in [("train", 60000)]:
        dataset = datasets.mnist(
            split_type=split, epochs=1, batch_size=batch_size, dataset_dir=DATASET_DIR,
        )

        for x, y in dataset:
            assert x.shape == (batch_size, 28, 28, 1)
            assert y.shape == (batch_size,)
            break
