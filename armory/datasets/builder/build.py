"""ARMORY Dataset Builder Script

This script is intended to be used to construct Armory Datasets from
the original data artifacts.  It can also be used to construct a dataset
from a local class_file + data artifacts using the `-lcs` options described
below.  The result is a `TFDS` style directory containing `tfrecord` files.

"""
import tensorflow_datasets as tfds
import os
import json
import shutil
from loguru import logger as log
import subprocess
import itertools
import armory.datasets.builder.utils as utils


def build_tfds_dataset(dataset_name: str, local_path: str, feature_dict=None) -> str:
    log.info(f"Building Dataset: {dataset_name} from TFDS artifacts")
    log.debug("Constructing Builder Object")
    builder = tfds.builder(dataset_name, data_dir=local_path)
    log.debug("Downloading artifacts...")
    builder.download_and_prepare()
    ds_path = utils.get_dataset_full_path(
        dataset_name, local_path, validate=False
    )  # Don't check structure yet
    log.debug(f"Resulting Dataset Directory: {ds_path}")
    if "features.json" not in os.listdir(ds_path):
        if feature_dict is None:
            raise RuntimeError(
                f"Cannot build TFDS Dataset for {dataset_name}...need `features_dict` specified"
                f"...consider adding `feature_dict` to SUPPORTED_DATASETS[{dataset_name}]"
            )
        else:
            log.info("Dataset `features.json` missing...writing to dataset")
            log.debug(f"contents: {feature_dict}")
            with open(os.path.join(ds_path, "features.json"), "w") as f:
                f.write(json.dumps(feature_dict))
    log.debug(f"Created Dataset: {ds_path} with contents: {os.listdir(ds_path)}")
    log.success(
        f"Dataset: {dataset_name} build Complete!!  Artifacts are located at {ds_path}"
    )
    return ds_path


def build_source_dataset(dataset_class_file: str, local_path: str) -> str:
    dataset_name = os.path.splitext(os.path.basename(dataset_class_file))[0]
    log.info(f"Building Dataset: {dataset_name} from source file: {dataset_class_file}")
    if not os.path.isfile(dataset_class_file):
        raise ValueError(
            f"Cannot build Dataset from source... source class file does not exist at: {dataset_class_file}"
        )

    cmd = f"tfds build {dataset_class_file} --data_dir {local_path}"
    log.info(f"Executing: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

    try:
        ds_path = utils.get_dataset_full_path(dataset_name, local_path)
    except (ValueError, RuntimeError) as e:
        log.error(
            "Something went wrong with tfds build... no dataset directory resulted"
        )
        raise e

    log.success(
        f"Dataset: {dataset_name} build Complete!!  Artifacts are located at {ds_path}"
    )
    return ds_path


def build(
    dataset_name: str, dataset_config, local_path: str, clean: bool = True
) -> str:
    """ Builds a given dataset from the provided inputs
    Parameters:
        dataset_name (str):         Name of the Dataset to build (e.g. `mnist`, `so2sat/all`, etc.)
        dataset_config (dict):      Configuration Dictonary for Dataset (see `utils.SUPPORTED_DATASETS`
                                     `utils.get_local_config` for details)
        local_path (str):           Output Location for the built dataset
        clean (bool):               Remove old artifacts and do "fresh" build of dataset
    """

    log.info(f"Building Dataset: {dataset_name}")
    log.debug(f"\t using Config: {dataset_config}")
    ds_path = os.path.join(
        local_path, dataset_name
    )  # Not sure it exists...don't use utils.get_ds_path
    log.trace(f"Initial ds_path: {ds_path}")
    if os.path.isdir(ds_path):
        log.info(f"Dataset: {dataset_name} already exists at {ds_path}")
        if not clean:
            log.warning("...skipping build. To overwrite use `--clean`")
            ds_path = utils.get_dataset_full_path(
                dataset_name, local_path
            )  # We want actual path with version
            log.debug(f"Returning ds_path: {ds_path}")
            return ds_path
        else:
            log.warning(f"Removing old dataset: {ds_path}!!")
            shutil.rmtree(ds_path)

    if dataset_config["type"] == "tfds":
        log.info("Generating Dataset from tfds artifacts!")
        ds_path = build_tfds_dataset(
            dataset_name,
            local_path=local_path,
            feature_dict=dataset_config["feature_dict"],
        )
        log.debug(f"Dataset Built at Path: {ds_path}")
        return ds_path

    elif dataset_config["type"] == "source":
        log.info("Generating Dataset from source!")
        ds_path = build_source_dataset(
            dataset_config["class_file"], local_path=local_path
        )
        log.debug(f"Dataset Built at Path: {ds_path}")
        return ds_path
    else:
        raise NotImplementedError(
            f"Supported Dataset {dataset_name}: is not implemented (or not valid)"
        )


if __name__ == "__main__":
    import argparse

    epilog = "\n".join(
        [
            "To Construct all datasets locally use:",
            "\t python build.py -ds all --clean",
            "or if you only want to build `mnist`, `digit, `cifar10`: ",
            "\t python build.py -ds mnist digit cifar10",
            "If you have a local class file and data at [my_data_dir] you can use:",
            "\t python build.py -lcs [my_data_dir] --clean",
            "\nNOTE: You must provide one and only one of `-ds`, `-lcs` `--list`.",
        ]
    )
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        usage="%(prog)s [options]",
        epilog=epilog,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-ds",
        "--dataset",
        choices=["all"] + list(utils.SUPPORTED_DATASETS.keys()),
        action="append",
        nargs="+",
        metavar="DATASET",
        default=None,
        help="Dataset name to generate",
    )
    group.add_argument(
        "-lcs",
        "--local-class-path",
        type=str,
        action="append",
        default=None,
        nargs="+",
        metavar="CLASS_FILE",
        help="Paths to files that contain TFDS builder classes",
    )
    group.add_argument(
        "--list",
        action="store_true",
        default=None,
        help="List SUPPORTED_DATASETS that can be built",
    )
    parser.add_argument(
        "--clean", action="store_true", help="Generate the dataset from scratch"
    )
    parser.add_argument(
        "-o",
        "--output-directory",
        default=utils.DEFAULT_DATASET_DIRECTORY,
        help="Directory to Store built datasets (Default: %(default)s )",
    )
    parser.add_argument(
        "-v",
        "--verbosity",
        type=str,
        choices=["trace", "debug", "info", "warning", "error"],
        default="info",
        help="Set Output log level (Default: %(default)s)",
    )
    args = parser.parse_args()

    # Setting up Logger to stdout with chosen level
    utils.setup_logger(
        level=args.verbosity.upper(),
        suppress_tfds_progress=True
        if args.verbosity in ["warning", "error"]
        else False,
    )

    # Listing SUPPORTED_DATASETS
    if args.list:
        print("Supported Datasets:\n\t" + "\n\t".join(utils.SUPPORTED_DATASETS.keys()))
        exit()

    if args.dataset is not None:
        args.dataset = list(itertools.chain(*args.dataset))  # Flatten list
        if "all" in args.dataset:
            dataset_dict = utils.SUPPORTED_DATASETS
        else:
            dataset_dict = {k: utils.SUPPORTED_DATASETS[k] for k in args.dataset}
    else:
        # Since mutually exclusive group...must have at least one local class path
        args.local_class_path = list(itertools.chain(*args.local_class_path))
        dataset_dict = {}
        for pyfile in args.local_class_path:
            dataset_name, dataset_config = utils.get_local_config(pyfile)
            dataset_dict[dataset_name] = dataset_config

    if len(dataset_dict.keys()) == 0:
        log.error("Must provide at least 1 dataset/class_file")
        raise ValueError("Must provide at least 1 dataset/class_file")

    log.info(f"Attempting to Build Datasets: {dataset_dict.keys()}")
    for ds_name, ds_config in dataset_dict.items():
        ds_path = build(ds_name, ds_config, args.output_directory, args.clean)
        try:
            ds_info, ds = utils.load(ds_name, args.output_directory)
        except Exception as e:
            log.exception(f"Could not reconstruct dataset located at {ds_path}!!")
            log.exception(e)
            raise e

    log.success("\t ALL Builds Complete !!")
