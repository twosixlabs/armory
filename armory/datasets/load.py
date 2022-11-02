import argparse

import tensorflow_datasets as tfds

from armory.datasets import common


def from_directory(builder_dir: str, **as_dataset_kwargs) -> (dict, dict):
    builder = tfds.builder_from_directory(builder_dir)
    ds = builder.as_dataset(**as_dataset_kwargs)
    return builder.info, ds


def load(name: str, version: str = None, **as_dataset_kwargs) -> (dict, dict):
    """
    Load the given dataset, optionally with specified version
        If version is not provided, the latest local version will be used

    By default, this loads from a base directory of ~/armory/dataset/new_builds

    Return (Info, Dataset) tuple
    """
    builder = tfds.builder(name, version=version, data_dir=common.get_root())
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
