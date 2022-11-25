import argparse

import tensorflow_datasets as tfds

from armory.datasets import common, download, package
from armory.logs import log


def from_directory(builder_dir: str, **as_dataset_kwargs) -> (dict, dict):
    builder = tfds.builder_from_directory(builder_dir)
    ds = builder.as_dataset(**as_dataset_kwargs)
    return builder.info, ds


def ensure_download_extract(
    name,
    version: str = None,
    data_dir: str = None,
    verify: bool = True,
    overwrite: bool = False,
    public: bool = True,
):
    try:
        info = common.cached_datasets()[name]
    except KeyError:
        raise KeyError(f"dataset {name} is not a cached dataset")

    version = info["version"]
    if version is not None and version != info["version"]:
        raise ValueError(
            f"provided dataset version {version} != {info['version']} cached version"
        )

    download.download(
        name, data_dir=data_dir, public=public, overwrite=overwrite, verify=verify
    )
    package.extract(name, data_dir=data_dir, overwrite=overwrite)


def load(
    name: str,
    version: str = None,
    data_dir: str = None,
    download_cached: bool = True,
    verify: bool = True,
    overwrite: bool = False,
    public: bool = True,
    **as_dataset_kwargs,
) -> (dict, dict):
    """
    Load the given dataset, optionally with specified version
        If version is not provided, the latest local version will be used

    By default, this loads from a base directory of ~/.armory/datasets/new_builds

    download_cached - if not present, download a cached version from s3, if it exists
    verify - whether to verify size and sha256 of download
    overwrite - whether to overwrite existings files with download and extraction
    public - whether to download from the public armory s3 bucket

    Return (Info, Dataset) tuple
    """
    if data_dir is None:
        data_dir = common.get_root()

    try:
        builder = tfds.builder(name, version=version, data_dir=data_dir)
        ds = builder.as_dataset(**as_dataset_kwargs)
        return builder.info, ds
    except (tfds.core.registered.DatasetNotFoundError, AssertionError):
        log.info(f"dataset {name} not found locally")
        if name in common.cached_datasets():
            if not download_cached:
                raise ValueError(
                    f"cached dataset {name} not found locally. set download_cached=True to download"
                )

            ensure_download_extract(
                name, version=version, verify=verify, overwrite=overwrite
            )
        elif name in common.armory_builders() or name in common.tfds_builders():
            raise ValueError(
                f"dataset {name} not cached. " "Please build via armory.datasets.build"
            )
        else:
            raise ValueError(f"dataset {name} not recognized")

    builder = tfds.builder(name, version=version, data_dir=data_dir)
    ds = builder.as_dataset(**as_dataset_kwargs)
    return builder.info, ds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "name",
        type=str,
        help="dataset name",
    )
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="specify the dataset version",
    )
    args = parser.parse_args()
    load(args.name, version=args.version)
