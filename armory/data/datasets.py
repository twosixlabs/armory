"""
Download and load baseline datasets with optional pre-processing.

Each baseline dataset resides in its own subdirectory under <dataset_dir> based
upon the name of the function in the datasets file. For example, the cifar10
data is found at '<dataset_dir>/cifar10'

The 'downloads' subdirectory under <dataset_dir> is reserved for caching.
"""

import logging
import os
from typing import Callable, Union

import numpy as np

# import torch before tensorflow to ensure torch.utils.data.DataLoader can utilize
#     all CPU resources when num_workers > 1
try:
    import torch  # noqa: F401
except ImportError:
    pass
import tensorflow as tf
import tensorflow_datasets as tfds
import apache_beam as beam
from art.data_generators import DataGenerator

from armory.data.utils import (
    download_verify_dataset_cache,
    _read_validate_scenario_config,
    add_checksums_dir,
)
from armory import paths
from armory.data.librispeech import librispeech_dev_clean_split  # noqa: F401
from armory.data.resisc45 import resisc45_split  # noqa: F401
from armory.data.xview import xview as xv  # noqa: F401
from armory.data.german_traffic_sign import german_traffic_sign as gtsrb  # noqa: F401
from armory.data.digit import digit as digit_tfds  # noqa: F401


os.environ["KMP_WARNINGS"] = "0"

logger = logging.getLogger(__name__)

CHECKSUMS_DIR = os.path.join(os.path.dirname(__file__), "url_checksums")
tfds.download.add_checksums_dir(CHECKSUMS_DIR)
CACHED_CHECKSUMS_DIR = os.path.join(os.path.dirname(__file__), "cached_s3_checksums")
add_checksums_dir(CACHED_CHECKSUMS_DIR)


class ArmoryDataGenerator(DataGenerator):
    """
    Returns batches of numpy data.

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

    @staticmethod
    def np_1D_object_array(x_list):
        """
        Take a list of single-element batches and return as a numpy 1D object array

        Similar to np.stack, but designed to handle variable-length elements
        """
        x = np.empty((len(x_list),), dtype=object)
        for i in range(len(x_list)):
            x[i] = x_list[i][0]
        return x

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
                if self.current == self.samples_per_epoch:
                    self.current = 0
                    break

            if isinstance(x_list[0], dict):
                # Translate a list of dicts into a dict of arrays
                x = {}
                for k in x_list[0].keys():
                    x[k] = self.np_1D_object_array([x_i[k] for x_i in x_list])
            elif isinstance(x_list[0], tuple):
                # Translate a list of tuples into a tuple of arrays
                x = tuple(self.np_1D_object_array(i) for i in zip(*x_list))
            else:
                x = self.np_1D_object_array(x_list)

            # Does not currently handle variable-length y
            if isinstance(y_list[0], dict):
                y = {}
                for k in y_list[0].keys():
                    y[k] = np.hstack([y_i[k] for y_i in y_list])
            elif isinstance(y_list[0], tuple):
                y = tuple(np.hstack(i) for i in zip(*y_list))
            else:
                y = np.hstack(y_list)
        else:
            x, y = next(self.generator)

        if self.preprocessing_fn:
            # Apply preprocessing to multiple inputs as needed
            if isinstance(x, dict):
                x = {k: self.preprocessing_fn(v) for (k, v) in x.items()}
            elif isinstance(x, tuple):
                x = tuple(self.preprocessing_fn(i) for i in x)
            else:
                x = self.preprocessing_fn(x)

        return x, y

    def __iter__(self):
        return self

    def __next__(self):
        return self.get_batch()

    def __len__(self):
        return self.batches_per_epoch * self.epochs


class EvalGenerator(DataGenerator):
    """
    Wraps a specified number of batches in a DataGenerator to allow for evaluating on
    part of a dataset when running through a scenario
    """

    def __init__(self, armory_generator, num_eval_batches):
        if not isinstance(armory_generator, ArmoryDataGenerator):
            raise ValueError(f"{armory_generator} is not of type ArmoryDataGenerator")
        super().__init__(armory_generator.batch_size, armory_generator.batch_size)
        self.armory_generator = armory_generator
        self.num_eval_batches = num_eval_batches
        self.batches_processed = 0
        # This attr is only used by ucf video scenarios that involve finetuning. It
        # must be set to enable check runs.
        self.batches_per_epoch = 1

    def get_batch(self) -> (np.ndarray, np.ndarray):
        if self.batches_processed == self.num_eval_batches:
            raise StopIteration()
        batch = self.armory_generator.get_batch()
        self.batches_processed += 1
        return batch

    def __iter__(self):
        return self

    def __next__(self):
        return self.get_batch()

    def __len__(self):
        return self.num_eval_batches


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
    framework: str = "numpy",
    lambda_map: Callable = None,
) -> Union[ArmoryDataGenerator, tf.data.Dataset]:
    """
    If as_supervised=False, must designate keys as a tuple in supervised_xy_keys:
        supervised_xy_keys=('video', 'label')  # ucf101 dataset
    if variable_length=True and batch_size > 1:
        output batches are 1D np.arrays of objects
    lambda_map - if not None, mapping function to apply to dataset elements
    """
    if not dataset_dir:
        dataset_dir = paths.runtime_paths().dataset_dir

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
        if not (isinstance(x_key, str) or isinstance(x_key, tuple)) or not isinstance(
            y_key, str
        ):
            raise ValueError(
                f"supervised_xy_keys must be a tuple of strings, or for x_key only, a tuple of tuple of strings"
                f" not {type(x_key), type(y_key)}"
            )
        if isinstance(x_key, tuple):
            for k in x_key:
                if not (isinstance(k, str)):
                    raise ValueError(
                        f"supervised_xy_keys must be a tuple of strings, or for x_key only, a tuple of tuple of strings"
                    )
            ds = ds.map(lambda x: (tuple(x[k] for k in x_key), x[y_key]))
        else:
            ds = ds.map(lambda x: (x[x_key], x[y_key]))
    if lambda_map is not None:
        ds = ds.map(lambda_map)

    ds = ds.repeat(epochs)
    if shuffle_files:
        ds = ds.shuffle(batch_size * 10, reshuffle_each_iteration=True)
    if variable_length and batch_size > 1:
        ds = ds.batch(1, drop_remainder=False)
    else:
        ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

    if framework == "numpy":
        ds = tfds.as_numpy(ds, graph=default_graph)
        generator = ArmoryDataGenerator(
            ds,
            size=ds_info.splits[split_type].num_examples,
            batch_size=batch_size,
            epochs=epochs,
            preprocessing_fn=preprocessing_fn,
            variable_length=bool(variable_length and batch_size > 1),
        )

    elif framework == "tf":
        generator = ds

    elif framework == "pytorch":
        torch_ds = _get_pytorch_dataset(ds)
        generator = torch.utils.data.DataLoader(
            torch_ds, batch_size=None, collate_fn=lambda x: x, num_workers=0
        )

    else:
        raise ValueError(
            f"`framework` must be one of ['tf', 'pytorch', 'numpy']. Found {framework}"
        )

    return generator


class MnistContext:
    def __init__(self):
        self.default_float = np.float32
        self.quantization = 255
        self.x_dimensions = (None, 28, 28, 1)


mnist_context = MnistContext()


def mnist_dataset_canonical_preprocessing(batch):
    if batch.ndim != len(mnist_context.x_dimensions):
        raise ValueError(
            f"input batch dim {batch.ndim} != {len(mnist_context.x_dimensions)}"
        )
    for dim, (source, target) in enumerate(
        zip(batch.shape, mnist_context.x_dimensions)
    ):
        pass
    assert batch.dtype == np.uint8
    assert batch.shape[1:] == mnist_context.x_dimensions[1:]

    batch = (
        batch.astype(mnist_context.default_float) / mnist_context.quantization
    )  # 255
    assert batch.dtype == mnist_context.default_float
    assert batch.max() <= 1.0
    assert batch.min() >= 0.0

    return batch


def mnist(
    split_type: str = "train",
    epochs: int = 1,
    batch_size: int = 1,
    dataset_dir: str = None,
    cache_dataset: bool = True,
    framework: str = "numpy",
    shuffle_files: bool = True,
) -> ArmoryDataGenerator:
    """
    Handwritten digits dataset:
        http://yann.lecun.com/exdb/mnist/
    """
    return _generator_from_tfds(
        "mnist:3.0.1",
        split_type=split_type,
        batch_size=batch_size,
        epochs=epochs,
        dataset_dir=dataset_dir,
        preprocessing_fn=mnist_dataset_canonical_preprocessing,
        cache_dataset=cache_dataset,
        framework=framework,
        shuffle_files=shuffle_files,
    )


def cifar10(
    split_type: str = "train",
    epochs: int = 1,
    batch_size: int = 1,
    dataset_dir: str = None,
    cache_dataset: bool = True,
    framework: str = "numpy",
    shuffle_files: bool = True,
) -> ArmoryDataGenerator:
    """
    Ten class image dataset:
        https://www.cs.toronto.edu/~kriz/cifar.html
    """
    return _generator_from_tfds(
        "cifar10:3.0.2",
        split_type=split_type,
        batch_size=batch_size,
        epochs=epochs,
        dataset_dir=dataset_dir,
        preprocessing_fn=cifar10_dataset_canonical_preprocessing,
        cache_dataset=cache_dataset,
        framework=framework,
        shuffle_files=shuffle_files,
    )


class Cifar10Context:
    def __init__(self):
        self.default_float = np.float32
        self.quantization = 255
        self.x_dimensions = (None, 32, 32, 3)


cifar10_context = Cifar10Context()


def cifar10_dataset_canonical_preprocessing(batch):
    if batch.ndim != len(cifar10_context.x_dimensions):
        raise ValueError(
            f"input batch dim {batch.ndim} != {len(cifar10_context.x_dimensions)}"
        )
    for dim, (source, target) in enumerate(
        zip(batch.shape, cifar10_context.x_dimensions)
    ):
        pass
    assert batch.dtype == np.uint8
    assert batch.shape[1:] == cifar10_context.x_dimensions[1:]

    batch = (
        batch.astype(cifar10_context.default_float) / cifar10_context.quantization
    )  # 255
    assert batch.dtype == cifar10_context.default_float
    assert batch.max() <= 1.0
    assert batch.min() >= 0.0

    return batch


def digit(
    split_type: str = "train",
    epochs: int = 1,
    batch_size: int = 1,
    dataset_dir: str = None,
    preprocessing_fn: Callable = None,
    cache_dataset: bool = True,
    framework: str = "numpy",
    shuffle_files: bool = True,
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
        framework=framework,
        shuffle_files=shuffle_files,
    )


def imagenette(
    split_type: str = "train",
    epochs: int = 1,
    batch_size: int = 1,
    dataset_dir: str = None,
    preprocessing_fn: Callable = None,
    cache_dataset: bool = True,
    framework: str = "numpy",
    shuffle_files: bool = True,
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
        framework=framework,
        shuffle_files=shuffle_files,
    )


def german_traffic_sign(
    split_type: str = "train",
    epochs: int = 1,
    batch_size: int = 1,
    preprocessing_fn: Callable = None,
    dataset_dir: str = None,
    cache_dataset: bool = True,
    framework: str = "numpy",
    shuffle_files: bool = True,
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
        framework=framework,
        shuffle_files=shuffle_files,
    )


def librispeech_dev_clean(
    split_type: str = "train",
    epochs: int = 1,
    batch_size: int = 1,
    dataset_dir: str = None,
    preprocessing_fn: Callable = None,
    cache_dataset: bool = True,
    framework: str = "numpy",
    shuffle_files: bool = True,
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
        framework=framework,
        shuffle_files=shuffle_files,
    )


def resisc45(
    split_type: str = "train",
    epochs: int = 1,
    batch_size: int = 1,
    dataset_dir: str = None,
    cache_dataset: bool = True,
    framework: str = "numpy",
    shuffle_files: bool = True,
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
        preprocessing_fn=resisc45_canonical_preprocessing,
        cache_dataset=cache_dataset,
        framework=framework,
        shuffle_files=shuffle_files,
    )


class Resisc45Context:
    def __init__(self):
        self.default_float = np.float32
        self.quantization = 255
        self.x_dimensions = (None, 256, 256, 3)


resisc45_context = Resisc45Context()


def resisc45_canonical_preprocessing(batch):
    if batch.ndim != len(resisc45_context.x_dimensions):
        raise ValueError(
            f"input batch dim {batch.ndim} != {len(resisc45_context.x_dimensions)}"
        )
    assert batch.dtype == np.uint8
    assert batch.shape[1:] == resisc45_context.x_dimensions[1:]

    batch = batch.astype(resisc45_context.default_float) / resisc45_context.quantization
    assert batch.dtype == resisc45_context.default_float
    assert batch.max() <= 1.0
    assert batch.min() >= 0.0

    return batch


def ucf101(
    split_type: str = "train",
    epochs: int = 1,
    batch_size: int = 1,
    dataset_dir: str = None,
    preprocessing_fn: Callable = None,
    cache_dataset: bool = True,
    framework: str = "numpy",
    shuffle_files: bool = True,
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
        framework=framework,
        shuffle_files=shuffle_files,
    )


def xview(
    split_type: str = "train",
    epochs: int = 1,
    batch_size: int = 1,
    dataset_dir: str = None,
    cache_dataset: bool = True,
    framework: str = "numpy",
    shuffle_files: bool = True,
) -> ArmoryDataGenerator:
    """
    split_type - one of ("train", "test")
    """
    return _generator_from_tfds(
        "xview:1.0.0",
        split_type=split_type,
        batch_size=batch_size,
        epochs=epochs,
        dataset_dir=dataset_dir,
        preprocessing_fn=xview_canonical_preprocessing,
        as_supervised=False,
        supervised_xy_keys=("image", "objects"),
        cache_dataset=cache_dataset,
        framework=framework,
        shuffle_files=shuffle_files,
    )


class XViewContext:
    def __init__(self):
        self.default_type = np.uint8
        self.x_dimensions = (
            None,
            None,
            None,
            3,
        )  # xview images are square but with different sizes


xview_context = XViewContext()


def xview_canonical_preprocessing(batch):
    if batch.ndim != len(xview_context.x_dimensions):
        raise ValueError(
            f"input batch dim {batch.ndim} != {len(xview_context.x_dimensions)}"
        )
    for dim, (source, target) in enumerate(
        zip(batch.shape, xview_context.x_dimensions)
    ):
        pass
    assert batch.dtype == xview_context.default_type
    assert batch.shape[1] == batch.shape[2]  # Ensure square shape
    assert batch.shape[3] == xview_context.x_dimensions[3]

    return batch


def so2sat(
    split_type: str = "train",
    epochs: int = 1,
    batch_size: int = 1,
    dataset_dir: str = None,
    cache_dataset: bool = True,
    framework: str = "numpy",
    shuffle_files: bool = True,
) -> ArmoryDataGenerator:
    return _generator_from_tfds(
        "so2sat/all:2.1.0",
        split_type=split_type,
        batch_size=batch_size,
        epochs=epochs,
        dataset_dir=dataset_dir,
        preprocessing_fn=None,
        as_supervised=False,
        supervised_xy_keys=(("sentinel1", "sentinel2"), "label"),
        lambda_map=so2sat_concat_map,
        cache_dataset=cache_dataset,
        framework=framework,
        shuffle_files=shuffle_files,
    )


def so2sat_concat_map(x, y):
    try:
        x1, x2 = x
    except (ValueError, TypeError):
        raise ValueError(
            "so2 dataset intermediate format corrupted. Should be in format (sentinel1,sentinel2),label"
        )
    return tf.concat([x1[..., :4], x2], -1), y


def _cache_dataset(dataset_dir: str, dataset_name: str):
    name, subpath = _parse_dataset_name(dataset_name)

    if not os.path.isdir(os.path.join(dataset_dir, name, subpath)):
        download_verify_dataset_cache(
            dataset_dir=dataset_dir, checksum_file=name + ".txt", name=name,
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
    "imagenette": imagenette,
    "german_traffic_sign": german_traffic_sign,
    "ucf101": ucf101,
    "resisc45": resisc45,
    "librispeech_dev_clean": librispeech_dev_clean,
    "xview": xview,
    "so2sat": so2sat,
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


def _get_pytorch_dataset(ds):
    import armory.data.pytorch_loader as ptl

    ds = ptl.TFToTorchGenerator(ds)

    return ds
