"""
Download and preprocess common datasets.
"""

import csv
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
    dataset_name: str, dataset_dir: str, preprocessing_fn: Callable,
) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    default_graph = tf.compat.v1.keras.backend.get_session().graph

    train_x, train_y = tfds.load(
        dataset_name,
        batch_size=-1,
        split="train",
        as_supervised=True,
        data_dir=dataset_dir,
    )

    test_x, test_y = tfds.load(
        dataset_name,
        batch_size=-1,
        split="test",
        as_supervised=True,
        data_dir=dataset_dir,
    )

    train_x = tfds.as_numpy(train_x, graph=default_graph)
    train_y = tfds.as_numpy(train_y, graph=default_graph)
    test_x = tfds.as_numpy(test_x, graph=default_graph)
    test_y = tfds.as_numpy(test_y, graph=default_graph)

    if preprocessing_fn:
        train_x = preprocessing_fn(train_x)
        test_x = preprocessing_fn(test_x)

    return train_x, train_y, test_x, test_y


def mnist(
    dataset_dir: str = None, preprocessing_fn: Callable = None,
) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Handwritten digits dataset:
        http://yann.lecun.com/exdb/mnist/

    returns:
        train_x, train_y, test_x, test_y
    """
    if not dataset_dir:
        dataset_dir = paths.docker().dataset_dir
    return _in_memory_dataset_tfds(
        "mnist:3.0.0", dataset_dir=dataset_dir, preprocessing_fn=preprocessing_fn
    )


def cifar10(
    dataset_dir: str = None, preprocessing_fn: Callable = None,
) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Ten class image dataset:
        https://www.cs.toronto.edu/~kriz/cifar.html

    returns:
        train_x, train_y, test_x, test_y
    """
    if not dataset_dir:
        dataset_dir = paths.docker().dataset_dir
    return _in_memory_dataset_tfds(
        "cifar10:3.0.0", dataset_dir=dataset_dir, preprocessing_fn=preprocessing_fn
    )


def digit(
    zero_pad: bool = False, dataset_dir: str = None,
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
    :param dataset_dir: Directory where cached datasets are stored
    :return: Train/Test arrays of audio and labels. Sample Rate is 8000 Hz
    """
    from scipy.io import wavfile

    if not dataset_dir:
        dataset_dir = paths.docker().dataset_dir

    rootdir = os.path.join(dataset_dir, "external")

    url = "https://github.com/Jakobovski/free-spoken-digit-dataset/archive/v1.0.8.zip"
    zip_file = "free-spoken-digit-dataset-1.0.8.zip"
    subdir = "free-spoken-digit-dataset-1.0.8/recordings"

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
    dataset_dir: str = None, preprocessing_fn: Callable = None,
) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    ILSVRC12 adversarial image dataset for ResNet50

    ProjectedGradientDescent
        Iterations = 10
        Max pertibation epsilon = 8
        Attack step size = 2
        Targeted = True

    :param dataset_dir: Directory where cached datasets are stored
    :param preprocessing_fn: Callable function to preprocess inputs
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

    if not dataset_dir:
        dataset_dir = paths.docker().dataset_dir

    num_images = 1000
    filename = "ILSVRC12_ResNet50_PGD_adversarial_dataset_v1.0.tfrecords"
    dirpath = os.path.join(dataset_dir, "external", "imagenet_adv")
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

    # Preprocessing should always be done on RGB inputs
    if preprocessing_fn:
        clean_x = preprocessing_fn(clean_x)
        adv_x = preprocessing_fn(adv_x)

    return clean_x, adv_x, labels


def german_traffic_sign(
    preprocessing_fn: Callable = None, dataset_dir: str = None,
) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    German traffic sign dataset with 43 classes and over
    50,000 images.

    :param preprocessing_fn: Callable function to preprocess inputs
    :param dataset_dir: Directory where cached datasets are stored
    :return: (train_images,train_labels,test_images,test_labels)
    """

    from PIL import Image

    def _read_images(prefix, gtFile, im_list, label_list):
        with open(gtFile, newline="") as csvFile:
            gtReader = csv.reader(
                csvFile, delimiter=";"
            )  # csv parser for annotations file
            gtReader.__next__()  # skip header
            # loop over all images in current annotations file
            for row in gtReader:
                try:
                    tmp = Image.open(os.path.join(prefix, row[0]))
                    # First column is filename
                except IOError as e:
                    raise IOError(f"Could not open image with PIL. {e}")
                im_list.append(np.array(tmp))
                tmp.close()
                label_list.append(int(row[7]))  # the 8th column is the label

    if not dataset_dir:
        dataset_dir = paths.docker().dataset_dir

    rootdir = os.path.join(dataset_dir, "external")
    subdir = "GTSRB"
    dirpath = os.path.join(rootdir, subdir)

    urls = [
        "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip",
        "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip",
        "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip",
    ]
    dirs = [rootdir, rootdir, dirpath]
    if not os.path.isdir(dirpath):
        for url, dir in zip(urls, dirs):
            zip_file = url.split("/")[-1]
            zip_filepath = os.path.join(dir, zip_file)
            # Download file if it does not exist
            if not os.path.isfile(zip_filepath):
                os.makedirs(dir, exist_ok=True)
                curl(url, dir, zip_file)

            # Extract and clean up
            with zipfile.ZipFile(zip_filepath, "r") as zip_ref:
                zip_ref.extractall(dir)
            os.remove(zip_filepath)

    train_images, train_labels = [], []
    test_images, test_labels = [], []

    for c in range(0, 43):
        prefix = os.path.join(
            dirpath, "Final_Training", "Images", format(c, "05d")
        )  # subdirectory for class
        gtFile = os.path.join(
            prefix, "GT-" + format(c, "05d") + ".csv"
        )  # annotations file
        _read_images(prefix, gtFile, train_images, train_labels)

    prefix = os.path.join(dirpath, "Final_Test", "Images")
    gtFile = os.path.join(dirpath, "GT-final_test.csv")
    _read_images(prefix, gtFile, test_images, test_labels)

    if preprocessing_fn:
        train_images = preprocessing_fn(train_images)
        test_images = preprocessing_fn(test_images)

    return (
        np.array(train_images),
        np.array(train_labels),
        np.array(test_images),
        np.array(test_labels),
    )


SUPPORTED_DATASETS = {
    "mnist": mnist,
    "cifar10": cifar10,
    "digit": digit,
    "imagenet_adversarial": imagenet_adversarial,
    "german_traffic_sign": german_traffic_sign,
}


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
