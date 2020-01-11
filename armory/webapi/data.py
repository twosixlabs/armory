"""
API queries to download and use common datasets.
"""

import os
import zipfile
import tensorflow as tf
import numpy as np


if tf.__version__[0] == "1":
    keras_backend = tf.keras.backend
elif tf.__version__[0] == "2":
    keras_backend = tf.compat.v1.keras.backend
else:
    raise ImportError(f"Requires TensorFlow 1 or 2, not {tf.__version__}")


def _curl(url, dirpath, filename):
    """
    git clone url from dirpath location
    """
    import subprocess

    try:
        subprocess.check_call(["curl", "-L", url, "--output", filename], cwd=dirpath)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"curl command not found. Is curl installed? {e}")
    except subprocess.CalledProcessError as e:
        raise subprocess.CalledProcessError(f"curl failed to download: {e}")


def _normalize_img_dataset(img, label):
    img = tf.cast(img, tf.float32) / 255.0
    return img, label


def _normalize_img(img):
    img = img.astype(np.float32) / 255.0
    return img


# TODO: Normalize is temporary until this is better refactored (Issue #10)
def mnist_data(batch_size: int = -1, epochs: int = 1, normalize: bool = False):
    """
    Returns (train_ds, (x_test, y_test), num_train, num_test)
        if batch_size == -1, train_ds = (x_train, y_train)
        else train_ds is a data generator

    :param batch_size:
    :param epochs: Number of times to repeat generator
    :return:
    """
    import tensorflow_datasets as tfds

    default_graph = keras_backend.get_session().graph

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
    if batch_size != -1 and epochs > 1:
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
def cifar10_data(batch_size: int = -1, epochs: int = 1, normalize: bool = False):
    """
    Returns (train_ds, (x_test, y_test), num_train, num_test)
        if batch_size == -1, train_ds = (x_train, y_train)
        else train_ds is a data generator

    :param batch_size:
    :param epochs: Number of times to repeat generator
    :return:
    """
    import tensorflow_datasets as tfds

    default_graph = keras_backend.get_session().graph

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
    if batch_size != -1 and epochs > 1:
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


def _create_generator(train_ds, batch_size, epochs):
    """
    Create generator from in-memory dataset.

    Keep leftover (partial-batch).
    """
    x_train, y_train = train_ds
    n = len(x_train)
    for epoch in range(epochs):
        for i in range(0, n, batch_size):
            yield x_train[i : i + batch_size], y_train[i : i + batch_size]


def digit(
    batch_size: int = -1, epochs: int = 1, zero_pad=False, rootdir="datasets/external"
):
    """
    returns:
        Return tuple of dictionaries containing numpy arrays.
        Keys are {`audio`, `label`}
        Sample Rate is 8000 Hz
        Audio samples are of different length, so this returns a numpy object array
            dtype of internal arrays are np.int16
            min length = 1148 samples
            max length = 18262 samples

    zero_pad - whether to pad the audio samples to the same length
        if done, this returns `audio` arrays as 2D np.int16 arrays

    rootdir - where the dataset is stored
    """
    import numpy as np
    from scipy.io import wavfile

    url = "https://github.com/Jakobovski/free-spoken-digit-dataset/archive/v1.0.8.zip"
    zip_file = "free-spoken-digit-dataset-1.0.8.zip"
    subdir = "free-spoken-digit-dataset-1.0.8/recordings"

    dirpath = os.path.join(rootdir, subdir)
    if not os.path.isdir(dirpath):
        zip_filepath = os.path.join(rootdir, zip_file)
        # Download file if it does not exist
        if not os.path.isfile(zip_filepath):
            os.makedirs(rootdir, exist_ok=True)
            _curl(url, rootdir, zip_file)

        # Extract and clean up
        with zipfile.ZipFile(zip_filepath, "r") as zip_ref:
            zip_ref.extractall(rootdir)
        os.remove(zip_filepath)

    # TODO: Return generators instead of all data in memory
    # NOTE: issue #21
    sample_rate = 8000
    max_length = 18262
    min_length = 1148
    dtype = np.int16
    train_audio, train_labels = [], []
    test_audio, test_labels = [], []
    for sample in range(50):
        for name in "jackson", "nicolas", "theo":  # , 'yweweler': not yet in release
            for digit in range(10):
                filepath = os.path.join(dirpath, f"{digit}_{name}_{sample}.wav")
                try:
                    s_r, audio = wavfile.read(filepath)
                except FileNotFoundError as e:
                    raise FileNotFoundError(f"digit dataset incomplete. {e}")
                if s_r != sample_rate:
                    raise ValueError(f"{filepath} sample rate {s_r} != {sample_rate}")
                if audio.dtype != dtype:
                    raise ValueError(f"{filepath} dtype {audio.dtype} != {dtype}")
                if zero_pad:
                    audio = np.hstack(
                        [audio, np.zeros(max_length - len(audio), dtype=np.int16)]
                    )
                if sample >= 5:
                    train_audio.append(audio)
                    train_labels.append(digit)
                else:
                    test_audio.append(audio)
                    test_labels.append(digit)

    train_ds = (np.array(train_audio), np.array(train_labels))
    test_ds = (np.array(test_audio), np.array(test_labels))
    # Generator
    if batch_size < -2 or batch_size == 0:
        raise ValueError(f"batch_size cannot be {batch_size}")
    batch_size = int(batch_size)
    epochs = int(epochs)
    if batch_size > 0:
        if epochs < 1:
            raise ValueError(f"epochs {epochs} must be >= 1 when batch_size != -1")
        train_ds = _create_generator(train_ds, batch_size, epochs)
    return train_ds, test_ds, len(train_audio), len(test_audio)


SUPPORTED_DATASETS = {"mnist": mnist_data, "cifar10": cifar10_data, "digit": digit}
