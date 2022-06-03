"""
Loader is used to return a TF Dataset object from a dataset
    It may handle partial loading and structuring via 'as_supervised'
"""

import os
from pathlib import Path
from typing import Optional, Tuple

from armory.logs import log
from armory.datasets.builder.utils import get_dataset_full_path
from armory.datasets import directory

import tensorflow_datasets as tfds


def supervised_keys_map(supervised, keys=None):
    """
    Return a function that maps from input dict to tuple output

    supervised - tuple of strings or tuple of tuple of strings (or Nones)
        Desired output format
        Examples:
            ("audio", "label")
            ("audio", "user_id")
            (("image", "adv_image"), "label")
            ("data", None)
    keys - set of allowable keys
        if None, do not check keys

    Input/Ouput Examples:
        ("hi", None, "there") --> lambda x: (x["hi"], None, x["there"])
        (("a", "b"), "c") --> lambda x: ((x["a"], x["b"]), x["c"])
    """

    def key_mapper(x):
        out = []
        for element in supervised:
            if element is None:
                out.append(None)
            elif isinstance(element, str):
                out.append(x[element])
            elif isinstance(element, tuple):
                out_sub = []
                for sub_element in element:
                    if sub_element is None:
                        out_sub.append(None)
                    elif isinstance(sub_element, str):
                        out_sub.append(x[sub_element])
                    else:
                        ValueError(f"sub_element {sub_element} must be None or str")
                out.append(tuple(out_sub))
            else:
                raise ValueError(f"element {element} must be None, str, or tuple")
        return tuple(out)

    if keys is not None:
        test_input = {k: i for i, k in enumerate(keys)}
        try:
            key_mapper(test_input)
        except KeyError as e:
            raise KeyError(f"element {e} is not in keys")

    return key_mapper


def from_config(
    name,
    version=None,
    split=None,
    supervised=True,
    batch_size=1,
    framework="numpy",
):
    info, ds = load_full(name, version - version, split=split, supervised=supervised)
    # TODO ...


def get_subdir(
    name,
    version=None,
    data_dir=None,
):
    """ """
    if version is None:  # derive version
        if name in directory.SUPPORTED_DATASETS:
            version = directory.SUPPORTED_DATASETS[name]
        elif name in tfds.list_builders():
            builder = tfds.builder(name)
            version = str(builder.version)
        else:
            raise ValueError(f"version not provided for custom dataset {name}")
    if data_dir is None:
        data_dir = paths.runtime_paths().dataset_dir
    data_dir = Path(data_dir)
    return data_dir / name / version


class Loader:
    def __init__(self, name, version=None, data_dir=None):
        self.supported = False
        if name in directory.SUPPORTED_DATASETS:
            self.supported = True
            supported_version, checksum = directory.SUPPORTED_DATASETS[name]
            if version is None:
                version = supported_version
            elif version != supported_version:
                log.warning(
                    f"version {version} does not match supported version {supported_version}"
                )
                self.supported = False
            self.checksum = checksum
        elif version is None:
            if name in tfds.list_builders():
                builder = tfds.builder(name)
                version = str(builder.version)
            else:
                raise ValueError(f"version not provided for custom dataset {name}")
        self.name = name
        self.version = version
        if data_dir is None:
            data_dir = paths.runtime_paths().dataset_dir
        self.data_dir = Path(data_dir)
        self.dataset_subdir = self.data_dir / name / version

    def load_builder(self, download_cached=True):
        if not self.dataset_subdir.is_dir():
            if download_cached:
                if checksum is None:
                    raise ValueError("No checksum for {self.name}. Cannot download")
                download.download(
                    name,
                    version,
                    self.data_dir,
                    checksum=self.checksum,
                    verify_download=True,
                )
                # TODO: unpack?

        self.builder = tfds.core.builder_from_directory(self.dataset_subdir)
        return self

    def set_supervised(self, supervised=False):
        supervised_map = None
        if supervised is True:
            as_supervised = True
        elif supervised is False or supervised is None:
            as_supervised = False
        elif isinstance(supervised, tuple) or isinstance(supervised, list):
            supervised = tuple(supervised)
            if split is None:
                raise NotImplementedError(
                    f"custom supervised keys require split specified"
                )
            as_supervised = False
            supervised_map = supervised_keys_map(supervised, keys=builder.info.features)
        else:
            ValueError(
                f"supervised must be one of (True, False, tuple), not {supervised}"
            )
        self.supervised_map, self.as_supervised = supervised_map, as_supervised

    def as_dataset(self, split=None, shuffle_files=None):
        ds = self.builder.as_dataset(
            split=split, shuffle_files=shuffle_files, as_supervised=self.as_supervised
        )
        if self.supervised_map is not None:
            ds = ds.map(self.supervised_map)
        self.ds = ds

    def get(self):
        return self.builder.info, ds


def load_full(
    name,
    *,
    # try_gcs = False,
    version=None,
    data_dir=None,
    download_cached=True,
    config=None,
    split=None,
    shuffle_files=False,
    supervised=False,
):
    """
    Return dataset info and dataset
    """
    path = get_subdir(name, version=version, data_dir=data_dir)
    if not path.is_dir():
        if download_cached and name in directory.SUPPORTED_DATASETS:
            download.download(name, version or directory.SUPPORTED_DATASETS[name])

        if path.is_dir():
            pass
    builder = tfds.core.builder_from_directory(path)

    if supervised is True:
        as_supervised = True
    elif supervised is False or supervised is None:
        as_supervised = False
    elif isinstance(supervised, tuple) or isinstance(supervised, list):
        supervised = tuple(supervised)
        if split is None:
            raise NotImplementedError(f"custom supervised keys require split specified")
        as_supervised = False
        supervised_map = supervised_keys_map(supervised, keys=builder.info.features)
    else:
        ValueError(f"supervised must be one of (True, False, tuple), not {supervised}")

    ds = builder.as_dataset(
        split=split, shuffle_files=shuffle_files, as_supervised=as_supervised
    )
    if isinstance(supervised, tuple):
        ds = ds.map(supervised_map)

    return builder.info, ds


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
