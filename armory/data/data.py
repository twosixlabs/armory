"""
Download and preprocess common datasets.
"""

import os
import zipfile
from typing import Callable

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from armory.data.utils import curl


def _in_memory_dataset_tfds(
    dataset_name: str, preprocessing_fn: Callable
) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    default_graph = tf.compat.v1.keras.backend.get_session().graph

    train_x, train_y = tfds.load(
        dataset_name,
        batch_size=-1,
        split="train",
        as_supervised=True,
        data_dir="datasets/",
    )

    test_x, test_y = tfds.load(
        dataset_name,
        batch_size=-1,
        split="test",
        as_supervised=True,
        data_dir="datasets/",
    )

    train_x = tfds.as_numpy(train_x, graph=default_graph)
    train_y = tfds.as_numpy(train_y, graph=default_graph)
    test_x = tfds.as_numpy(test_x, graph=default_graph)
    test_y = tfds.as_numpy(test_y, graph=default_graph)

    if preprocessing_fn:
        train_x = preprocessing_fn(train_x)
        test_x = preprocessing_fn(test_x)

    return train_x, train_y, test_x, test_y


def mnist_data(
    preprocessing_fn: Callable = None,
) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Handwritten digits dataset:
        http://yann.lecun.com/exdb/mnist/

    returns:
        train_x, train_y, test_x, test_y
    """
    return _in_memory_dataset_tfds("mnist", preprocessing_fn=preprocessing_fn)


def cifar10_data(
    preprocessing_fn: Callable = None,
) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Ten class image dataset:
        https://www.cs.toronto.edu/~kriz/cifar.html

    returns:
        train_x, train_y, test_x, test_y
    """
    return _in_memory_dataset_tfds("cifar10", preprocessing_fn=preprocessing_fn)


def digit(
    zero_pad: bool = False, rootdir: str = "datasets/external",
) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    An audio dataset of spoken digits:
        https://github.com/Jakobovski/free-spoken-digit-dataset

    Audio samples are of different length, so this returns a numpy object array
            dtype of internal arrays are np.int16
            min length = 1148 samples
            max length = 18262 samples

    :param zero_pad: Boolean to pad the audio samples to the same length
        if `True`, this returns `audio` arrays as 2D np.int16 arrays
    :param rootdir: Directory where the dataset is stored
    :return: Train/Test arrays of audio and labels. Sample Rate is 8000 Hz
    """
    from scipy.io import wavfile

    url = "https://github.com/Jakobovski/free-spoken-digit-dataset/archive/v1.0.8.zip"
    zip_file = "free-spoken-digit-dataset-1.0.8.zip"
    subdir = "free-spoken-digit-dataset-1.0.8/recordings"

    # TODO: Refactor as a decorator for external downloads
    dirpath = os.path.join(rootdir, subdir)
    if not os.path.isdir(dirpath):
        zip_filepath = os.path.join(rootdir, zip_file)
        # Download file if it does not exist
        if not os.path.isfile(zip_filepath):
            os.makedirs(rootdir, exist_ok=True)
            curl(url, rootdir, zip_file)

        # Extract and clean up
        with zipfile.ZipFile(zip_filepath, "r") as zip_ref:
            zip_ref.extractall(rootdir)
        os.remove(zip_filepath)

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

    return (
        np.array(train_audio),
        np.array(train_labels),
        np.array(test_audio),
        np.array(test_labels),
    )


SUPPORTED_DATASETS = {"mnist": mnist_data, "cifar10": cifar10_data, "digit": digit}
