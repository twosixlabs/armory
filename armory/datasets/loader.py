from armory.logs import log
import tensorflow_datasets as tfds
from armory.datasets.builder.utils import get_dataset_full_path
import os
from pathlib import Path
from .generator import ArmoryDataGenerator
from typing import Union
import tensorflow as tf
import tensorflow_datasets as tfds
import torch

from typing import Callable, Optional


def load_from_directory(
    dataset_full_path: str, as_supervised: bool = True
) -> (dict, dict):
    log.info(f"Attempting to Load Dataset from local directory: {dataset_full_path}")
    log.debug("Generating Builder object...")
    builder = tfds.core.builder_from_directory(dataset_full_path)
    log.debug(f"Dataset Full Name: `{builder.info.full_name}`")
    ds = builder.as_dataset(as_supervised=as_supervised)
    log.success("Loading Complete!!")
    return builder.info, ds


def load(dataset_name: str, dataset_directory: str, as_supervised: bool = True):
    """Loads the TFDS Dataset using `tfds.core.builder_from_directory` method
    Parameters:
        dataset_name (str):         Name of the Dataset/Directory (e.g. `mnist`)
        dataset_directory (str):    Directory containing the Dataset

    Returns:
        ds_info (obj):              The TFDS dataset info JSON object.
        ds (tfds.core.Dataset)      The TFDS dataset
    """

    ds_path = get_dataset_full_path(dataset_name, dataset_directory)
    expected_name = f"{Path(ds_path).relative_to(Path(dataset_directory))}"
    log.debug(
        f"ds_path: {ds_path}, expected_name: {expected_name}, dataset_directory: {dataset_directory}"
    )
    if not os.path.isdir(ds_path):
        raise ValueError(
            f"Dataset Directory: {ds_path} does not exist...cannot construct!!"
        )

    ds_info, ds = load_from_directory(ds_path, as_supervised)

    if expected_name != ds_info.full_name:
        raise RuntimeError(
            f"Dataset Full Name: {ds_info.full_name}  differs from expected: {expected_name}"
            "...make sure that the build_class_file name matches the class name!!"
            "NOTE:  tfds converts camel case class names to lowercase separated by `_`"
        )
    log.debug("Converting to dataset")

    return ds_info, ds


def generator_from_dataset(
    dataset_info: dict,
    dataset: dict,
    framework: str = "numpy",
    split: str = "train",
    batch_size: int = 1,
    epochs: int = 1,
    preprocessing_fn: Callable = None,
    label_preprocessing_fn: Callable = None,
    variable_length: bool = False,
    variable_y: bool = False,
    shuffle_files: bool = False,
    as_supervised: bool = True,
    supervised_xy_keys: Optional[tuple] = None,
    lambda_map: Callable = None,
    context=None,
    class_ids=None,
    index=None,
) -> Union[ArmoryDataGenerator, tf.data.Dataset]:

    ds = dataset[split]

    if not as_supervised:
        try:
            x_key, y_key = supervised_xy_keys
        except (TypeError, ValueError):
            raise ValueError(
                f"When as_supervised=False, supervised_xy_keys must be a (x_key, y_key)"
                f" tuple, not {supervised_xy_keys}"
            )
        for key in [x_key, y_key]:
            if not (isinstance(key, str) or isinstance(key, tuple)):
                raise ValueError(
                    f"supervised_xy_keys must be a tuple of strings or a tuple of tuple of strings"
                    f" not {type(x_key), type(y_key)}"
                )
        if isinstance(x_key, tuple):
            if isinstance(y_key, tuple):
                raise ValueError(
                    "Only one of (x_key, y_key) can be a tuple while the other must be a string."
                )
            for k in x_key:
                if not (isinstance(k, str)):
                    raise ValueError(
                        "supervised_xy_keys must be a tuple of strings or a tuple of tuple of strings"
                    )
            ds = ds.map(lambda x: (tuple(x[k] for k in x_key), x[y_key]))
        elif isinstance(y_key, tuple):
            for k in y_key:
                if not (isinstance(k, str)):
                    raise ValueError(
                        "supervised_xy_keys must be a tuple of strings or a tuple of tuple of strings"
                    )
            ds = ds.map(lambda x: (x[x_key], tuple(x[k] for k in y_key)))
        else:
            ds = ds.map(lambda x: (x[x_key], x[y_key]))

    # TODO:  Remove , thesea are just notes
    #  this shows up in so2stat call
    if lambda_map is not None:
        ds = ds.map(lambda_map)

    dataset_size = dataset_info.splits[split].num_examples

    # Add class-based filtering
    if class_ids is not None:
        if split == "train":
            log.warning(
                "Filtering by class entails iterating over the whole dataset and thus "
                "can be very slow if using the 'train' split"
            )

        # TODO: Why not use TFDS filter ?? (https://www.tensorflow.org/datasets/decode#only_decode_a_sub-set_of_the_features)

        # TODO: Remove when done
        #  Issue with filtering, to know the len you have to iterate the entire dataset, doesn't appear in metadata
        #  Filter by index is fast, filter by stringslice is very slow
        #  Figure out why we need the full dataset size
        #  Add to ArmoryDataGenerator -> add_filter that removes samples at execution time based on filter
        if isinstance(class_ids, list):
            ds, dataset_size = filter_by_class(ds, class_ids=class_ids)
        elif isinstance(class_ids, int):
            ds, dataset_size = filter_by_class(ds, class_ids=[class_ids])
        else:
            raise ValueError(
                f"class_ids must be a list, int, or None, not {type(class_ids)}"
            )

    # Add index-based filtering
    if isinstance(index, list):
        ds, dataset_size = filter_by_index(ds, index, dataset_size)
    elif isinstance(index, str):
        ds, dataset_size = filter_by_str_slice(ds, index, dataset_size)
    elif index is not None:
        raise ValueError(f"index must be a list, str, or None, not {type(index)}")

    ds = ds.repeat(epochs)
    # TODO: Why is this here since builder does this already??
    #  shuffle files is a part of original builder but not during execution
    #  maybe not needed if we shuffle the files at build time
    if shuffle_files:
        ds = ds.shuffle(batch_size * 10, reshuffle_each_iteration=True)
    if variable_length or variable_y and batch_size > 1:
        ds = ds.batch(1, drop_remainder=False)
    else:
        ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

    log.info(f"ds: {ds}")

    if framework == "numpy":
        ds = tfds.as_numpy(ds)
        log.debug(f"Numpy ds: {ds}")
        generator = ArmoryDataGenerator(
            iter(ds),
            size=dataset_size,
            batch_size=batch_size,
            epochs=epochs,
            preprocessing_fn=preprocessing_fn,
            label_preprocessing_fn=label_preprocessing_fn,
            variable_length=bool(variable_length and batch_size > 1),
            variable_y=bool(variable_y and batch_size > 1),
            context=context,
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

    log.debug(f"generator: {generator}")
    return generator
