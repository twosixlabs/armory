"""
Loader is used to return a TF Dataset object from a dataset
    It may handle partial loading and structuring via 'as_supervised'
"""

import os
from pathlib import Path

from armory.logs import log
from armory.datasets.builder.utils import get_dataset_full_path

import tensorflow_datasets as tfds


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
