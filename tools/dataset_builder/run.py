import tensorflow_datasets as tfds
import os
import json
import shutil
import sys
from loguru import logger as log
import subprocess

SUPPORTED_DATASETS = {
    "mnist": {
        "type": "tfds",
        "feature_dict": {
            "type": "tensorflow_datasets.core.features.features_dict.FeaturesDict",
            "content": json.dumps(
                {
                    "features": {
                        "image": {
                            "pythonClassName": "tensorflow_datasets.core.features.image_feature.Image",
                            "image": {
                                "shape": {"dimensions": ["-1", "-1", "1"]},
                                "dtype": "uint8",
                            },
                        },
                        "label": {
                            "pythonClassName": "tensorflow_datasets.core.features.class_label_feature.ClassLabel",
                            "classLabel": {"numClasses": "10"},
                        },
                    }
                }
            ),
            "proto_cls": "tensorflow_datasets.FeaturesDict",
        },
    },
    "cifar10": {"type": "tfds", "feature_dict": None},
    "cifar100": {"type": "tfds", "feature_dict": None},
    "digit": {"type": "source", "version": "1.0.8"},
}


def build_tfds_dataset(dataset_name: str, local_path: str) -> str:
    log.info(f"Building Dataset: {dataset_name} from TFDS artifacts")
    log.debug("Constructing Builder Object")
    builder = tfds.builder(dataset_name, data_dir=local_path)
    log.debug("Downloading artifacts...")
    builder.download_and_prepare()
    versions = next(os.walk(os.path.join(local_path, ds)))[1]
    log.debug(f"Got Versions: {versions} (expected to be len == 1)")
    assert len(versions) == 1
    ds_path = os.path.join(local_path, dataset_name, versions[0])
    if "features.json" not in os.listdir(ds_path):
        if SUPPORTED_DATASETS[dataset_name]["feature_dict"] is None:
            raise RuntimeError(
                f"Cannot build TFDS Dataset for {dataset_name}...need `features_dict` specified"
                f"...consider adding `feature_dict` to SUPPORTED_DATASETS[{dataset_name}]"
            )
        else:
            log.info("Dataset `features.json` missing...writing to dataset")
            log.debug(f"contents: {SUPPORTED_DATASETS[dataset_name]['feature_dict']}")
            with open(os.path.join(ds_path, "features.json"), "w") as f:
                f.write(json.dumps(SUPPORTED_DATASETS[dataset_name]["feature_dict"]))
    log.success(
        f"Dataset: {dataset_name} build Complete!!  Artifacts are located at {ds_path}"
    )
    return ds_path


def build_source_dataset(dataset_name: str, local_path: str) -> str:
    # TODO refactor to use python api instead of subprocess
    log.info(f"Building Dataset: {dataset_name} from source")
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Cannot build Dataset: {dataset_name} from source...consider updating `SUPPORTED_DATASETS`"
        )

    pth = os.path.join(os.path.dirname(__file__), dataset_name, f"{dataset_name}.py")
    if not os.path.isfile(pth):
        raise ValueError(
            f"Cannot build Dataset: {dataset_name} from source... source class does not exist at: {pth}"
        )

    cmd = f"tfds build {pth} --data_dir {local_path}"
    log.info(f"Executing: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

    ds_path = os.path.join(local_path, dataset_name)
    if not os.path.isdir(ds_path):
        raise RuntimeError(
            f"Something went wrong with tfds build....no resulting dataset directory {ds_path}"
        )

    versions = next(os.walk(ds_path))[1]
    log.debug(f"Got Versions: {versions} (expected to be len == 1)")
    assert len(versions) == 1
    ds_path = os.path.join(ds_path, versions[0])
    log.success(
        f"Dataset: {dataset_name} build Complete!!  Artifacts are located at {ds_path}"
    )
    return ds_path


def build(dataset_name: str, local_path: str, clean: bool = True) -> str:
    log.info(f"Constructing Dataset: {dataset_name}")
    ds_path = os.path.join(local_path, dataset_name)
    if os.path.isdir(ds_path):
        log.info(f"Dataset: {dataset_name} already exists at {ds_path}")
        if not clean:
            log.warning("...skipping build. To overwrite use `--clean`")
            return ds_path
        else:
            log.warning(f"Removing old dataset: {ds_path}!!")
            shutil.rmtree(ds_path)

    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(f"Dataset: {dataset_name} is not supported!!!")

    if SUPPORTED_DATASETS[dataset_name]["type"] == "tfds":
        log.info("Generating Dataset from tfds artifacts!")
        ds_path = build_tfds_dataset(dataset_name, local_path=local_path)
        return ds_path

    elif SUPPORTED_DATASETS[dataset_name]["type"] == "source":
        log.info("Generating Dataset from source!")
        ds_path = build_source_dataset(dataset_name, local_path=local_path)
        return ds_path
    else:
        raise NotImplementedError(
            f"Supported Dataset Type: {SUPPORTED_DATASETS[dataset_name]} is not implemented"
        )


def construct(dataset_directory: str):
    if not os.path.isdir(dataset_directory):
        raise ValueError(
            f"Dataset Directory: {dataset_directory} does not exist...cannot construct!!"
        )

    log.info("Attempting to Construct Dataset from local artifacts")
    log.debug("Generating Builder object...")
    builder = tfds.core.builder_from_directory(dataset_directory)
    log.debug("Converting to dataset")
    ds = builder.as_dataset()
    log.success("Construction Complete!!")
    return builder.info, ds


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        choices=["all"] + list(SUPPORTED_DATASETS.keys()),
        help="Dataset name to generate",
    )
    parser.add_argument(
        "--clean", action="store_true", help="Generate the dataset from scratch"
    )
    parser.add_argument(
        "-o",
        "--output-directory",
        default=os.path.expanduser(os.path.join("~", ".armory", "dataset_builds")),
        help="Directory to Store built datasets",
    )
    parser.add_argument(
        "-v",
        "--verbosity",
        type=str,
        choices=["trace", "debug", "info", "warning", "error"],
        default="info",
        help="Set Output log level",
    )
    args = parser.parse_args()

    # Setting up Logger to stdout with chosen level
    log.remove()
    log.add(sys.stdout, level=args.verbosity.upper())

    args.dataset = (
        list(SUPPORTED_DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    )
    for ds in args.dataset:
        ds_path = build(ds, args.output_directory, args.clean)
        try:
            ds_info, ds = construct(ds_path)
        except Exception as e:
            log.exception(f"Could not reconstruct dataset located at {ds_path}!!")
            log.exception(e)
