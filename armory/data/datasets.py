"""
Download and load baseline datasets with optional pre-processing.

Each baseline dataset resides in its own subdirectory under <dataset_dir> based
upon the name of the function in the datasets file. For example, the cifar10
data is found at '<dataset_dir>/cifar10'

The 'downloads' subdirectory under <dataset_dir> is reserved for caching.
"""

import logging
import os
from typing import Callable

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import apache_beam as beam

from art.data_generators import DataGenerator
from armory.data.utils import (
    download_verify_dataset_cache,
    _read_validate_scenario_config,
)
from armory import paths
from armory.data.librispeech import librispeech_dev_clean_split  # noqa: F401
from armory.data.resisc45 import resisc45_split  # noqa: F401
from armory.data.german_traffic_sign import german_traffic_sign as gtsrb  # noqa: F401
from armory.data.adversarial import imagenet_adversarial as IA  # noqa: F401
from armory.data.digit import digit as digit_tfds  # noqa: F401


os.environ["KMP_WARNINGS"] = "0"

logger = logging.getLogger(__name__)

CHECKSUMS_DIR = os.path.join(os.path.dirname(__file__), "url_checksums")
tfds.download.add_checksums_dir(CHECKSUMS_DIR)
CACHED_CHECKSUMS_DIR = os.path.join(os.path.dirname(__file__), "cached_s3_checksums")


class ArmoryDataGenerator(DataGenerator):
    """
    Returns batches of data.

    variable_length - if True, returns a 1D object array of arrays for x.
    """

    def __init__(
        self,
        generator,
        size,
        epochs,
        batch_size,
        preprocessing_fn=None,
        variable_length=False,
    ):
        super().__init__(size, batch_size)
        self.preprocessing_fn = preprocessing_fn
        self.generator = generator

        self.epochs = epochs
        self.samples_per_epoch = size

        # drop_remainder is False
        self.batches_per_epoch = self.samples_per_epoch // batch_size + bool(
            self.samples_per_epoch % batch_size
        )

        self.variable_length = variable_length
        if self.variable_length:
            self.current = 0

    def get_batch(self) -> (np.ndarray, np.ndarray):
        if self.variable_length:
            # build the batch
            x_list, y_list = [], []
            for i in range(self.batch_size):
                x_i, y_i = next(self.generator)
                x_list.append(x_i[0])
                y_list.append(y_i)
                self.current += 1
                # handle end of epoch partial batches
                if self.current == self.samples_per_epoch:
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

    def __iter__(self):
        return self

    def __next__(self):
        return self.get_batch()

    def __len__(self):
        return self.batches_per_epoch * self.epochs


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
    shuffle_files=True,
    cache_dataset: bool = True,
):
    """
    If as_supervised=False, must designate keys as a tuple in supervised_xy_keys:
        supervised_xy_keys=('video', 'label')  # ucf101 dataset
    if variable_length=True and batch_size > 1:
        output batches are 1D np.arrays of objects
    """
    if not dataset_dir:
        dataset_dir = paths.docker().dataset_dir

    if cache_dataset:
        _cache_dataset(
            dataset_dir, dataset_name=dataset_name,
        )

    default_graph = tf.compat.v1.keras.backend.get_session().graph

    ds, ds_info = tfds.load(
        dataset_name,
        split=split_type,
        as_supervised=as_supervised,
        data_dir=dataset_dir,
        with_info=True,
        download_and_prepare_kwargs=download_and_prepare_kwargs,
        shuffle_files=shuffle_files,
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
    if shuffle_files:
        ds = ds.shuffle(batch_size * 10, reshuffle_each_iteration=True)
    if variable_length and batch_size > 1:
        ds = ds.batch(1, drop_remainder=False)
    else:
        ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    ds = tfds.as_numpy(ds, graph=default_graph)

    generator = ArmoryDataGenerator(
        ds,
        size=ds_info.splits[split_type].num_examples,
        batch_size=batch_size,
        epochs=epochs,
        preprocessing_fn=preprocessing_fn,
        variable_length=bool(variable_length and batch_size > 1),
    )

    return generator


def mnist(
    split_type: str = "train",
    epochs: int = 1,
    batch_size: int = 1,
    dataset_dir: str = None,
    preprocessing_fn: Callable = None,
    cache_dataset: bool = True,
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
        cache_dataset=cache_dataset,
    )


def cifar10(
    split_type: str = "train",
    epochs: int = 1,
    batch_size: int = 1,
    dataset_dir: str = None,
    preprocessing_fn: Callable = None,
    cache_dataset: bool = True,
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
        cache_dataset=cache_dataset,
    )


def digit(
    split_type: str = "train",
    epochs: int = 1,
    batch_size: int = 1,
    dataset_dir: str = None,
    preprocessing_fn: Callable = None,
    cache_dataset: bool = True,
) -> ArmoryDataGenerator:
    """
    An audio dataset of spoken digits:
        https://github.com/Jakobovski/free-spoken-digit-dataset
    """
    return _generator_from_tfds(
        "digit:1.0.8",
        split_type=split_type,
        batch_size=batch_size,
        epochs=epochs,
        dataset_dir=dataset_dir,
        preprocessing_fn=preprocessing_fn,
        variable_length=bool(batch_size > 1),
        cache_dataset=cache_dataset,
    )


def imagenet_adversarial(
    split_type: str = "clean",
    epochs: int = 1,
    batch_size: int = 1,
    dataset_dir: str = None,
    preprocessing_fn: Callable = None,
    cache_dataset: bool = True,
) -> ArmoryDataGenerator:
    """
    ILSVRC12 adversarial image dataset for ResNet50

    ProjectedGradientDescent
        Iterations = 10
        Max perturbation epsilon = 8
        Attack step size = 2
        Targeted = True
    """

    return _generator_from_tfds(
        "imagenet_adversarial:1.0.0",
        split_type=split_type,
        batch_size=batch_size,
        epochs=epochs,
        dataset_dir=dataset_dir,
        preprocessing_fn=preprocessing_fn,
        shuffle_files=False,
        cache_dataset=cache_dataset,
    )


def imagenette(
    split_type: str = "train",
    epochs: int = 1,
    batch_size: int = 1,
    dataset_dir: str = None,
    preprocessing_fn: Callable = None,
    cache_dataset: bool = True,
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
        cache_dataset=cache_dataset,
    )


def german_traffic_sign(
    split_type: str = "train",
    epochs: int = 1,
    batch_size: int = 1,
    preprocessing_fn: Callable = None,
    dataset_dir: str = None,
    cache_dataset: bool = True,
) -> ArmoryDataGenerator:
    """
    German traffic sign dataset with 43 classes and over 50,000 images.
    """
    return _generator_from_tfds(
        "german_traffic_sign:3.0.0",
        split_type=split_type,
        batch_size=batch_size,
        epochs=epochs,
        dataset_dir=dataset_dir,
        preprocessing_fn=preprocessing_fn,
        variable_length=bool(batch_size > 1),
        cache_dataset=cache_dataset,
    )


def librispeech_dev_clean(
    split_type: str = "train",
    epochs: int = 1,
    batch_size: int = 1,
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

    return _generator_from_tfds(
        "librispeech_dev_clean_split/plain_text:1.1.0",
        split_type=split_type,
        batch_size=batch_size,
        epochs=epochs,
        dataset_dir=dataset_dir,
        preprocessing_fn=preprocessing_fn,
        download_and_prepare_kwargs={"download_config": dl_config},
        variable_length=bool(batch_size > 1),
        cache_dataset=cache_dataset,
    )


def resisc45(
    split_type: str = "train",
    epochs: int = 1,
    batch_size: int = 1,
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
    return _generator_from_tfds(
        "resisc45_split:3.0.0",
        split_type=split_type,
        batch_size=batch_size,
        epochs=epochs,
        dataset_dir=dataset_dir,
        preprocessing_fn=preprocessing_fn,
        cache_dataset=cache_dataset,
    )


def ucf101(
    split_type: str = "train",
    epochs: int = 1,
    batch_size: int = 1,
    dataset_dir: str = None,
    preprocessing_fn: Callable = None,
    cache_dataset: bool = True,
) -> ArmoryDataGenerator:
    """
    UCF 101 Action Recognition Dataset
        https://www.crcv.ucf.edu/data/UCF101.php
    """

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
        cache_dataset=cache_dataset,
    )


def _cache_dataset(dataset_dir: str, dataset_name: str):
    name, subpath = _parse_dataset_name(dataset_name)

    if not os.path.isdir(os.path.join(dataset_dir, name, subpath)):
        download_verify_dataset_cache(
            dataset_dir=dataset_dir,
            checksum_file=os.path.join(CACHED_CHECKSUMS_DIR, name + ".txt"),
            name=name,
        )


def _parse_dataset_name(dataset_name: str):
    try:
        name_config, version = dataset_name.split(":")
        splits = name_config.split("/")
        if len(splits) > 2:
            raise ValueError
        name = splits[0]
        config = splits[1:]
        subpath = os.path.join(*config + [version])
    except ValueError:
        raise ValueError(
            f'Dataset name "{dataset_name}" not properly formatted.\n'
            'Should be formatted "<name>[/<config>]:<version>", '
            'where "[]" indicates "/<config>" is optional.'
        )
    return name, subpath


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


def download_all(download_config, scenario):
    """
    Download all datasets for a scenario or requested datset to cache.
    """

    def _print_scenario_names():
        logger.info(
            f"The following scenarios are available based upon config file {download_config}:"
        )
        for scenario in config["scenario"].keys():
            logger.info(scenario)

    config = _read_validate_scenario_config(download_config)
    if scenario == "all":
        for scenario in config["scenario"].keys():
            for dataset in config["scenario"][scenario]["dataset_name"]:
                _download_data(dataset)
    elif scenario == "list":
        _print_scenario_names()
    else:
        if scenario not in config["scenario"].keys():
            logger.info(f"The scenario name {scenario} is not valid.")
            _print_scenario_names()
            raise ValueError("Invalid scenario name.")

        for dataset in config["scenario"][scenario]["dataset_name"]:
            _download_data(dataset)


def _download_data(dataset_name):
    """
    Download a single dataset to cache.
    """
    if dataset_name not in SUPPORTED_DATASETS.keys():
        raise ValueError(
            f"dataset {dataset_name} not supported. Must be one of {list(SUPPORTED_DATASETS.keys())}"
        )

    func = SUPPORTED_DATASETS[dataset_name]

    logger.info(f"Downloading (if necessary) dataset {dataset_name}...")

    try:
        func()
        logger.info(f"Successfully downloaded dataset {dataset_name}.")
    except Exception:
        logger.exception(f"Loading dataset {dataset_name} failed.")
