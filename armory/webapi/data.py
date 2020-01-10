"""
API queries to download and use common datasets.
"""
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


def _normalize_img_dataset(img, label):
    img = tf.cast(img, tf.float32) / 255.0
    return img, label


def _normalize_img(img):
    img = img.astype(np.float32) / 255.0
    return img


# TODO: Normalize is temporary until this is better refactored (Issue #10)
def mnist_data(batch_size: int, epochs: int, normalize: bool=False):
    """
    Tuple of dictionaries containing numpy arrays. Keys are {`image`, `label`}

    :param batch_size:
    :param epochs: Number of times to repeat generator
    :return:
    """
    default_graph = tf.keras.backend.get_session().graph

    mnist_builder = tfds.builder("mnist")
    num_train = mnist_builder.info.splits["train"].num_examples
    num_test = mnist_builder.info.splits["test"].num_examples

    train_ds = tfds.load(
        "mnist",
        split="train",
        batch_size=batch_size,
        as_supervised=True,
        data_dir="datasets/",
    )
    if normalize:
        train_ds = train_ds.map(_normalize_img_dataset)
    train_ds = train_ds.repeat(epochs)
    train_ds = tfds.as_numpy(train_ds, graph=default_graph)

    # TODO: Make generator once ART accepts generators in attack/defense methods (Issue #13)
    test_ds = tfds.load(
        "mnist", split="test", batch_size=-1, as_supervised=True, data_dir="datasets/"
    )
    test_x, test_y = tfds.as_numpy(test_ds, graph=default_graph)
    if normalize:
        test_x = _normalize_img(test_x)

    return train_ds, (test_x, test_y), num_train, num_test

# TODO: Normalize is temporary until this is better refactored (Issue #10)
def cifar10_data(batch_size: int, epochs: int, normalize: bool=False):
    """
    Tuple of dictionaries containing numpy arrays. Keys are {`image`, `label`}

    :param batch_size:
    :param epochs: Number of times to repeat generator
    :return:
    """
    default_graph = tf.keras.backend.get_session().graph

    mnist_builder = tfds.builder("cifar10")
    num_train = mnist_builder.info.splits["train"].num_examples
    num_test = mnist_builder.info.splits["test"].num_examples

    train_ds = tfds.load(
        "cifar10",
        split="train",
        batch_size=batch_size,
        as_supervised=True,
        data_dir="datasets/",
    )

    if normalize:
        train_ds = train_ds.map(_normalize_img_dataset)

    train_ds = train_ds.repeat(epochs)
    train_ds = tfds.as_numpy(train_ds, graph=default_graph)

    # TODO: Make generator once ART accepts generators in attack/defense methods (Issue #13)
    test_ds = tfds.load(
        "cifar10", split="test", batch_size=-1, as_supervised=True, data_dir="datasets/"
    )
    test_x, test_y = tfds.as_numpy(test_ds, graph=default_graph)
    if normalize:
        test_x = _normalize_img(test_x)

    return train_ds, (test_x, test_y), num_train, num_test


SUPPORTED_DATASETS = {"mnist": mnist_data, "cifar10": cifar10_data}
