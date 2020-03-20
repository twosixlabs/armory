"""
Download and preprocess common datasets.
Each standard dataset resides in its own subdirectory under <dataset_dir> based
upon the name of the function in the datasets file. For example, the cifar10
data is found at '<dataset_dir>/cifar10'
The 'download' subdirectory under <dataset_dir> is reserved for caching.
The 'private' subdirectory under <dataset_dir> is reserved for private
datasets.
"""

import csv
import logging
import os
import zipfile
from typing import Callable

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import apache_beam as beam

from art.data_generators import DataGenerator
from armory.data.utils import curl, download_file_from_s3, download_verify_dataset_cache
from armory import paths
from armory.data.librispeech import librispeech_dev_clean_split  # noqa: F401
from armory.data.resisc45 import resisc45_split  # noqa: F401


os.environ["KMP_WARNINGS"] = "0"

logger = logging.getLogger(__name__)

CHECKSUMS_DIR = os.path.join(os.path.dirname(__file__), "url_checksums")
tfds.download.add_checksums_dir(CHECKSUMS_DIR)
CACHED_CHECKSUMS_DIR = os.path.join(os.path.dirname(__file__), "cached_url_checksums")


class ArmoryDataGenerator(DataGenerator):
    """
    Returns batches of data.

    variable_length - if True, returns a 1D object array of arrays for x.
    """

    def __init__(
        self, generator, size, batch_size, preprocessing_fn=None, variable_length=False
    ):
        super().__init__(size, batch_size)
        self.preprocessing_fn = preprocessing_fn
        self.generator = generator
        # drop_remainder is False
        self.total_iterations = size // batch_size + bool(size % batch_size)
        self.variable_length = variable_length
        if self.variable_length:
            self.current = 0

    def get_batch(self) -> (np.ndarray, np.ndarray):
        if self.variable_length:
            # build the batch
            x_list, y_list = [], []
            for i in range(self.batch_size):
                x_i, y_i = next(self.generator)
                x_list.append(x_i)
                y_list.append(y_i)
                self.current += 1
                # handle end of epoch partial batches
                if self.current == self.size:
                    self.current = 0
                    break
            x = np.empty((len(x_list),), dtype=object)
            for i in range(len(x_list)):
                x[i] = x_list[i]
            # only handles variable-length x, currently
            y = np.hstack(y_list)
        else:
            x, y = next(self.generator)

        if self.preprocessing_fn:
            x = self.preprocessing_fn(x)

        return x, y


def _generator_from_tfds(
    dataset_name: str,
    split_type: str,
    batch_size: int,
    epochs: int,
    dataset_dir: str,
    preprocessing_fn: Callable,
    as_supervised: bool = True,
    supervised_xy_keys=None,
    download_and_prepare_kwargs=None,
    variable_length=False,
):
    """
    If as_supervised=False, must designate keys as a tuple in supervised_xy_keys:
        supervised_xy_keys=('video', 'label')  # ucf101 dataset
    if variable_length=True and batch_size > 1:
        output batches are 1D np.arrays of objects
    """
    if not dataset_dir:
        dataset_dir = paths.docker().dataset_dir

    default_graph = tf.compat.v1.keras.backend.get_session().graph

    ds, ds_info = tfds.load(
        dataset_name,
        split=split_type,
        as_supervised=as_supervised,
        data_dir=dataset_dir,
        with_info=True,
        download_and_prepare_kwargs=download_and_prepare_kwargs,
        shuffle_files=True,
    )
    if not as_supervised:
        try:
            x_key, y_key = supervised_xy_keys
        except (TypeError, ValueError):
            raise ValueError(
                f"When as_supervised=False, supervised_xy_keys must be a (x_key, y_key)"
                f" tuple, not {supervised_xy_keys}"
            )
        if not isinstance(x_key, str) or not isinstance(y_key, str):
            raise ValueError(
                f"supervised_xy_keys be a tuple of strings,"
                f" not {type(x_key), type(y_key)}"
            )
        ds = ds.map(lambda x: (x[x_key], x[y_key]))

    ds = ds.repeat(epochs)
    ds = ds.shuffle(batch_size * 10, reshuffle_each_iteration=True)
    if variable_length and batch_size > 1:
        ds = ds.batch(1, drop_remainder=False)
    else:
        ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    ds = tfds.as_numpy(ds, graph=default_graph)

    generator = ArmoryDataGenerator(
        ds,
        size=epochs * ds_info.splits[split_type].num_examples,
        batch_size=batch_size,
        preprocessing_fn=preprocessing_fn,
        variable_length=bool(variable_length and batch_size > 1),
    )

    return generator


def _inner_generator(
    X: list,
    Y: list,
    batch_size: int,
    epochs: int,
    drop_remainder: bool = False,
    variable_length=False,
):
    """
    Create a generator from lists (or arrays) of numpy arrays
    """
    num_examples = len(X)
    batch_size = int(batch_size)
    epochs = int(epochs)
    if len(X) != len(Y):
        raise ValueError("X and Y must have the same length")
    if epochs < 0:
        raise ValueError("epochs cannot be negative")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    if drop_remainder:
        num_examples = (num_examples // batch_size) * batch_size
    if num_examples == 0:
        return

    Z = list(zip(X, Y))
    for epoch in range(epochs):
        np.random.shuffle(Z)
        for start in range(0, num_examples, batch_size):
            x_list, y_list = zip(*Z[start : start + batch_size])
            if variable_length:
                x = np.empty((len(x_list),), dtype=object)
                for i in range(len(x_list)):
                    x[i] = x_list[i]
            else:
                x = np.stack(x_list)
            yield x, np.stack(y_list)


def _generator_from_np(
    X: list,
    Y: list,
    batch_size: int,
    epochs: int,
    preprocessing_fn: Callable,
    variable_length: bool = False,
) -> ArmoryDataGenerator:
    """
    Create generator from (X, Y) lists numpy arrays
    """
    ds = _inner_generator(
        X, Y, batch_size, epochs, drop_remainder=False, variable_length=variable_length
    )

    # variable_length not needed for ArmoryDataGenerator because np handles it
    return ArmoryDataGenerator(
        ds,
        size=epochs * len(X),
        batch_size=batch_size,
        preprocessing_fn=preprocessing_fn,
    )


def mnist(
    split_type: str,
    epochs: int,
    batch_size: int,
    dataset_dir: str = None,
    preprocessing_fn: Callable = None,
) -> ArmoryDataGenerator:
    """
    Handwritten digits dataset:
        http://yann.lecun.com/exdb/mnist/
    """
    return _generator_from_tfds(
        "mnist:3.0.0",
        split_type=split_type,
        batch_size=batch_size,
        epochs=epochs,
        dataset_dir=dataset_dir,
        preprocessing_fn=preprocessing_fn,
    )


def cifar10(
    split_type: str,
    epochs: int,
    batch_size: int,
    dataset_dir: str = None,
    preprocessing_fn: Callable = None,
) -> ArmoryDataGenerator:
    """
    Ten class image dataset:
        https://www.cs.toronto.edu/~kriz/cifar.html
    """
    return _generator_from_tfds(
        "cifar10:3.0.0",
        split_type=split_type,
        batch_size=batch_size,
        epochs=epochs,
        dataset_dir=dataset_dir,
        preprocessing_fn=preprocessing_fn,
    )


def digit(
    split_type: str,
    epochs: int,
    batch_size: int,
    dataset_dir: str = None,
    preprocessing_fn: Callable = None,
    zero_pad: bool = False,
) -> ArmoryDataGenerator:
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

    variable_length = not zero_pad and batch_size > 1

    if split_type == "train":
        samples = range(5, 50)
    elif split_type == "test":
        samples = range(5)
    else:
        raise ValueError(f"split_type {split_type} must be one of ('train', 'test')")

    rootdir = os.path.join(dataset_dir, "digit")

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
    audio_list, labels = [], []
    for sample in samples:
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
                audio_list.append(audio)
                labels.append(digit)

    return _generator_from_np(
        audio_list,
        labels,
        batch_size,
        epochs,
        preprocessing_fn,
        variable_length=variable_length,
    )


def imagenet_adversarial(
    batch_size: int,
    epochs: int,
    split_type: str,
    dataset_dir: str = None,
    preprocessing_fn: Callable = None,
) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    ILSVRC12 adversarial image dataset for ResNet50

    ProjectedGradientDescent
        Iterations = 10
        Max perturbation epsilon = 8
        Attack step size = 2
        Targeted = True

    :param dataset_dir: Directory where cached datasets are stored
    :param preprocessing_fn: Callable function to preprocess inputs
    :return: (Adversarial_images, Labels)
    """

    def _parse(serialized_example, split_type):
        ds_features = {
            "height": tf.io.FixedLenFeature([], tf.int64),
            "width": tf.io.FixedLenFeature([], tf.int64),
            "label": tf.io.FixedLenFeature([], tf.int64),
            "adv-image": tf.io.FixedLenFeature([], tf.string),
            "clean-image": tf.io.FixedLenFeature([], tf.string),
        }

        example = tf.io.parse_single_example(serialized_example, ds_features)

        if split_type == "clean":
            img = tf.io.decode_raw(example["clean-image"], tf.float32)
            img = tf.reshape(img, (example["height"], example["width"], -1))
        elif split_type == "adversarial":
            img = tf.io.decode_raw(example["adv-image"], tf.float32)
            img = tf.reshape(img, (example["height"], example["width"], -1))

        label = tf.cast(example["label"], tf.int32)
        return img, label

    default_graph = tf.compat.v1.keras.backend.get_session().graph

    acceptable_splits = ["clean", "adversarial"]
    if split_type not in acceptable_splits:
        raise ValueError(f"split_type must be one of {acceptable_splits}")

    if not dataset_dir:
        dataset_dir = paths.docker().dataset_dir

    num_images = 1000
    filename = "ILSVRC12_ResNet50_PGD_adversarial_dataset_v1.0.tfrecords"
    dirpath = os.path.join(dataset_dir, "imagenet_adversarial", "imagenet_adv")
    output_filepath = os.path.join(dirpath, filename)

    os.makedirs(dirpath, exist_ok=True)
    download_file_from_s3(
        bucket_name="armory-public-data",
        key=f"imagenet-adv/{filename}",
        local_path=output_filepath,
    )

    ds = tf.data.TFRecordDataset(filenames=[output_filepath])
    ds = ds.map(lambda example_proto: _parse(example_proto, split_type))
    ds = ds.repeat(epochs)
    ds = ds.shuffle(batch_size * 10)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    ds = tfds.as_numpy(ds, graph=default_graph)

    generator = ArmoryDataGenerator(
        ds,
        size=epochs * num_images,
        batch_size=batch_size,
        preprocessing_fn=preprocessing_fn,
    )

    return generator


def imagenette(
    split_type: str,
    epochs: int,
    batch_size: int,
    dataset_dir: str = None,
    preprocessing_fn: Callable = None,
) -> ArmoryDataGenerator:
    """
    Smaller subset of 10 classes of Imagenet
        https://github.com/fastai/imagenette
    """

    return _generator_from_tfds(
        "imagenette/full-size:0.1.0",
        split_type=split_type,
        batch_size=batch_size,
        epochs=epochs,
        dataset_dir=dataset_dir,
        preprocessing_fn=preprocessing_fn,
        variable_length=bool(batch_size > 1),
    )


def german_traffic_sign(
    split_type: str,
    epochs: int,
    batch_size: int,
    preprocessing_fn: Callable = None,
    dataset_dir: str = None,
) -> ArmoryDataGenerator:
    """
    German traffic sign dataset with 43 classes and over
    50,000 images.

    :param preprocessing_fn: Callable function to preprocess inputs
    :param dataset_dir: Directory where cached datasets are stored
    :return: generator
    """

    if split_type not in ["train", "test"]:
        raise ValueError(
            f"Split value of {split_type} is invalid for German traffic sign dataset. Must be one of 'train' or 'test'."
        )
    variable_length = bool(batch_size > 1)

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

    rootdir = os.path.join(dataset_dir, "german_traffic_sign")
    subdir = "GTSRB"
    dirpath = os.path.join(rootdir, subdir)
    num_classes = 43

    # Download all data on the first call regardless of split
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
    images, labels = [], []
    if split_type == "train":
        for c in range(0, num_classes):
            prefix = os.path.join(
                dirpath, "Final_Training", "Images", format(c, "05d")
            )  # subdirectory for class
            gtFile = os.path.join(
                prefix, "GT-" + format(c, "05d") + ".csv"
            )  # annotations file
            _read_images(prefix, gtFile, images, labels)
    else:  # split_type == "test"
        prefix = os.path.join(dirpath, "Final_Test", "Images")
        gtFile = os.path.join(dirpath, "GT-final_test.csv")
        _read_images(prefix, gtFile, images, labels)
    return _generator_from_np(
        images,
        labels,
        batch_size,
        epochs,
        preprocessing_fn,
        variable_length=variable_length,
    )


def librispeech_dev_clean(
    split_type: str,
    epochs: int,
    batch_size: int,
    dataset_dir: str = None,
    preprocessing_fn: Callable = None,
    cache_dataset: bool = True,
):
    """
    Librispeech dev dataset with custom split used for speaker
    identification

    split_type - one of ("train", "validation", "test")

    returns:
        Generator
    """
    flags = []
    dl_config = tfds.download.DownloadConfig(
        beam_options=beam.options.pipeline_options.PipelineOptions(flags=flags)
    )

    if cache_dataset:
        _cache_dataset(
            dataset_dir,
            name="librispeech_dev_clean_split",
            subpath=os.path.join("plain_text", "1.1.0"),
        )

    return _generator_from_tfds(
        "librispeech_dev_clean_split:1.1.0",
        split_type=split_type,
        batch_size=batch_size,
        epochs=epochs,
        dataset_dir=dataset_dir,
        preprocessing_fn=preprocessing_fn,
        download_and_prepare_kwargs={"download_config": dl_config},
        variable_length=bool(batch_size > 1),
    )


def resisc45(
    split_type: str,
    epochs: int,
    batch_size: int,
    dataset_dir: str = None,
    preprocessing_fn: Callable = None,
    cache_dataset: bool = True,
) -> ArmoryDataGenerator:
    """
    REmote Sensing Image Scene Classification (RESISC) dataset
        http://http://www.escience.cn/people/JunweiHan/NWPU-RESISC45.html

    Contains 31,500 images covering 45 scene classes with 700 images per class

    Uses TFDS:
        https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/image/resisc45.py

    Dimensions of X: (31500, 256, 256, 3) of uint8, ~ 5.8 GB in memory
        Each sample is a 256 x 256 3-color (RGB) image
    Dimensions of y: (31500,) of int, with values in range(45)

    split_type - one of ("train", "validation", "test")
    """
    if cache_dataset:
        _cache_dataset(
            dataset_dir, name="resisc45_split", subpath="3.0.0",
        )

    return _generator_from_tfds(
        "resisc45_split:3.0.0",
        split_type=split_type,
        batch_size=batch_size,
        epochs=epochs,
        dataset_dir=dataset_dir,
        preprocessing_fn=preprocessing_fn,
    )


def ucf101(
    split_type: str,
    epochs: int,
    batch_size: int = 1,
    dataset_dir: str = None,
    preprocessing_fn: Callable = None,
    cache_dataset: bool = True,
) -> ArmoryDataGenerator:
    """
    UCF 101 Action Recognition Dataset
        https://www.crcv.ucf.edu/data/UCF101.php
    """

    if cache_dataset:
        _cache_dataset(
            dataset_dir, name="ucf101", subpath=os.path.join("ucf101_1", "2.0.0"),
        )

    return _generator_from_tfds(
        "ucf101/ucf101_1:2.0.0",
        split_type=split_type,
        batch_size=batch_size,
        epochs=epochs,
        dataset_dir=dataset_dir,
        preprocessing_fn=preprocessing_fn,
        as_supervised=False,
        supervised_xy_keys=("video", "label"),
        variable_length=bool(batch_size > 1),
    )


def _cache_dataset(dataset_dir: str, name: str, subpath: str):
    if not dataset_dir:
        dataset_dir = paths.docker().dataset_dir

    if not os.path.isdir(os.path.join(dataset_dir, name, subpath)):
        download_verify_dataset_cache(
            dataset_dir=dataset_dir,
            checksum_file=os.path.join(CACHED_CHECKSUMS_DIR, name + ".txt"),
            name=name,
        )


SUPPORTED_DATASETS = {
    "mnist": mnist,
    "cifar10": cifar10,
    "digit": digit,
    "imagenet_adversarial": imagenet_adversarial,
    "imagenette": imagenette,
    "german_traffic_sign": german_traffic_sign,
    "ucf101": ucf101,
    "resisc45": resisc45,
    "librispeech_dev_clean": librispeech_dev_clean,
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
