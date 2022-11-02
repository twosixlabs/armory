import argparse
from pathlib import Path
import subprocess

import tensorflow_datasets as tfds

from armory.datasets import common
from armory.logs import log


def build_tfds_dataset(name: str, data_dir: str = None, overwrite: bool = False):
    """
    Wrapper around `tfds build` - builds a single tfds dataset from raw files
        Return target subdirectory
    """
    if data_dir is None:
        data_dir = common.get_root()

    cmd = [
        "tfds",
        "build",
        name,
        "--data_dir",
        f"{data_dir}",
        "--force_checksums_validation",
    ]
    if overwrite:
        cmd.append("--overwrite")
    print(f"Executing: {' '.join(cmd)}")
    # TODO: if this command fails, why isn't a CalledProcessError raised?
    # https://docs.python.org/3/library/subprocess.html
    subprocess.run(cmd, check=True)

    build = tfds.builder(name, data_dir=data_dir)
    return build.info.data_dir  # full data subdirectory


def build_armory_dataset(
    name: str,
    data_dir: str = None,
    overwrite: bool = False,
    register_checksums: bool = False,
):
    """
    Wrapper around `tfds build` - builds a single armory tfds dataset from source
        Return target subdirectory  # TODO: currently returns completed_process
    """
    if data_dir is None:
        data_dir = common.get_root()

    source_subdir = armory_builders()[name]

    cmd = [
        "tfds",
        "build",
        f"{source_subdir}",
        "--data_dir",
        f"{data_dir}",
    ]
    if register_checksums:
        cmd.append("--register_checksums")
    else:
        cmd.append("--force_checksums_validation")
    if overwrite:
        cmd.append("--overwrite")
    print(f"Executing: {' '.join(cmd)}")
    # TODO: if this command fails, why isn't a CalledProcessError raised?
    # https://docs.python.org/3/library/subprocess.html
    subprocess.run(cmd, check=True)

    build = tfds.builder(name, data_dir=data_dir)
    return build.info.data_dir  # full data subdirectory


def build_custom_dataset(
    name: str,
    data_dir: str = None,
    overwrite: bool = False,
    register_checksums: bool = False,
):
    raise NotImplementedError


def build(
    name: str,
    data_dir: str = None,
    overwrite: bool = False,
    register_checksums: bool = False,
):
    """
    Build the given dataset

    name - name of the dataset to build (e.g., 'mnist', 'digit')

    overwrite - whether to remove old artifacts and do a "fresh" build of dataset

    register_checksums - generate checksums file; only used for armory and custom datasets
        Always set to False for TFDS built-in datasets

    If register_checksums is False, then force_checksums_validation is set to True

    By default, this builds in a base directory of <armory_dataset_dir>/new_builds

    Return the subdirectory of the built dataset
    """
    if name in common.armory_builders():
        # build via armory
        return build_armory_dataset(
            name,
            data_dir=data_dir,
            overwrite=overwrite,
            register_checksums=register_checksums,
        )
    elif name in common.tfds_builders():
        return build_tfds_dataset(name, data_dir=data_dir, overwrite=overwrite)
    else:
        return build_custom_dataset(
            name,
            data_dir=data_dir,
            overwrite=overwrite,
            register_checksums=register_checksums,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "name",
        type=str,
        help="dataset name",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="whether to remove old artifacts before building",
    )
    parser.add_argument(
        "--register_checksums",
        action="store_true",
        help="whether to populate 'checksums.tsv' file for dataset",
    )
    args = parser.parse_args()
    build(
        args.name, overwrite=args.overwrite, register_checksums=args.register_checksums
    )
