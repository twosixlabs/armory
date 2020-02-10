"""
Download and preprocess common datasets.
"""

import logging
import os
import zipfile
from typing import Callable

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from armory.data.utils import curl, download_file_from_s3
from armory import paths

os.environ["KMP_WARNINGS"] = "0"


logger = logging.getLogger(__name__)


def _in_memory_dataset_tfds(
    dataset_name: str, preprocessing_fn: Callable
) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    default_graph = tf.compat.v1.keras.backend.get_session().graph

    train_x, train_y = tfds.load(
        dataset_name,
        batch_size=-1,
        split="train",
        as_supervised=True,
        data_dir=paths.DATASETS,
    )

    test_x, test_y = tfds.load(
        dataset_name,
        batch_size=-1,
        split="test",
        as_supervised=True,
        data_dir=paths.DATASETS,
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
    return _in_memory_dataset_tfds("mnist:3.0.0", preprocessing_fn=preprocessing_fn)


def cifar10_data(
    preprocessing_fn: Callable = None,
) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Ten class image dataset:
        https://www.cs.toronto.edu/~kriz/cifar.html

    returns:
        train_x, train_y, test_x, test_y
    """
    return _in_memory_dataset_tfds("cifar10:3.0.0", preprocessing_fn=preprocessing_fn)


def digit(
    zero_pad: bool = False, rootdir: str = os.path.join(paths.DATASETS, "external"),
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
                if not (min_length <= len(audio) <= max_length):
                    raise ValueError(f"{filepath} audio length {len(audio)}")
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


def imagenet_adversarial(
    preprocessing_fn: Callable = None,
    dirpath: str = os.path.join(paths.DATASETS, "external", "imagenet_adv"),
) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    ILSVRC12 adversarial image dataset for ResNet50

    ProjectedGradientDescent
        Iterations = 10
        Max pertibation epsilon = 8
        Attack step size = 2
        Targeted = True

    :param preprocessing_fn: Callable function to preprocess inputs
    :param dirpath: Directory where the dataset is stored
    :return: (Adversarial_images, Labels)
    """

    def _parse(serialized_example):
        ds_features = {
            "height": tf.io.FixedLenFeature([], tf.int64),
            "width": tf.io.FixedLenFeature([], tf.int64),
            "label": tf.io.FixedLenFeature([], tf.int64),
            "adv-image": tf.io.FixedLenFeature([], tf.string),
            "clean-image": tf.io.FixedLenFeature([], tf.string),
        }

        example = tf.io.parse_single_example(serialized_example, ds_features)

        clean_img = tf.io.decode_raw(example["clean-image"], tf.float32)
        clean_img = tf.reshape(clean_img, (example["height"], example["width"], -1))

        adv_img = tf.io.decode_raw(example["adv-image"], tf.float32)
        adv_img = tf.reshape(adv_img, (example["height"], example["width"], -1))

        label = tf.cast(example["label"], tf.int32)
        return clean_img, adv_img, label

    num_images = 1000
    filename = "ILSVRC12_ResNet50_PGD_adversarial_dataset_v0.1.tfrecords"
    output_filepath = os.path.join(dirpath, filename)

    os.makedirs(dirpath, exist_ok=True)
    download_file_from_s3(
        bucket_name="armory-public-data",
        key=f"imagenet-adv/{filename}",
        local_path=output_filepath,
    )

    adv_ds = tf.data.TFRecordDataset(filenames=[output_filepath])
    image_label_ds = adv_ds.map(lambda example_proto: _parse(example_proto))

    image_label_ds = image_label_ds.batch(num_images)
    image_label_ds = tf.data.experimental.get_single_element(image_label_ds)
    clean_x, adv_x, labels = tfds.as_numpy(image_label_ds)

    # Temporary flip from BGR to RGB since dataset was saved in BGR.
    clean_x = clean_x[..., ::-1]
    adv_x = adv_x[..., ::-1]

    # Preprocessing should always be done on RGB inputs
    if preprocessing_fn:
        clean_x = preprocessing_fn(clean_x)
        adv_x = preprocessing_fn(adv_x)

    return clean_x, adv_x, labels


SUPPORTED_DATASETS = {
    "mnist": mnist_data,
    "cifar10": cifar10_data,
    "digit": digit,
    "imagenet_adversarial": imagenet_adversarial,
}


def load(name, *args, **kwargs):
    """
    Return dataset or raise KeyError

    Convenience function, essentially.
    """
    dataset_function = SUPPORTED_DATASETS.get(name)
    if dataset_function is None:
        raise KeyError(
            f"{name} is not in supported datasets: {SUPPORTED_DATASETS.keys()}"
        )
    return dataset_function(*args, **kwargs)


def download_all():
    """
    Download all datasets to cache.
    """
    errors = []
    for name, func in SUPPORTED_DATASETS.items():
        logger.info(f"Downloading (if necessary) dataset {name}")
        try:
            func()
        except Exception:
            errors.append(name)
            logger.exception(f"Loading dataset {name} failed.")
    if errors:
        logger.info("All datasets downloaded successfully")
    else:
        logger.error(f"The following datasets failed to download: {errors}")
