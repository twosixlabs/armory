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
import sys
from loguru import logger as log
import subprocess
import itertools
import tools

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
    "imagenette/full-size": {"type": "tfds", "feature_dict": None},
    "coco/2017": {"type": "tfds", "feature_dict": None},
    "so2sat/all": {"type": "tfds", "feature_dict": None},
    "carla_obj_det_train": {
        "type": "source",
        "class_file": os.path.join(
            os.path.dirname(__file__), "build_classes", "carla_obj_det_train.py"
        ),
    },
    "digit": {
        "type": "source",
        "class_file": os.path.join(
            os.path.dirname(__file__), "build_classes", "digit.py"
        ),
    },
    "german_traffic_sign": {
        "type": "source",
        "class_file": os.path.join(
            os.path.dirname(__file__), "build_classes", "german_traffic_sign.py"
        ),
    },
    # TODO Add Librispeech (structure seems strange, used deprecated builder, and errors with connection timeout)
    # "librispeech_full": {
    #     "type": "source",
    #     "class_file": os.path.join(os.path.dirname(__file__), "build_classes", "librispeech_full.py"),
    # },
    # "librispeech_dev_clean_split": {
    #     "type": "source",
    #     "class_file": os.path.join(os.path.dirname(__file__), "build_classes", "librispeech_dev_clean_split.py"),
    # },
    "resisc10_poison": {
        "type": "source",
        "class_file": os.path.join(
            os.path.dirname(__file__), "build_classes", "resisc10_poison.py"
        ),
    },
    "resisc45_split": {
        "type": "source",
        "class_file": os.path.join(
            os.path.dirname(__file__), "build_classes", "resisc45_split.py"
        ),
    },
    # TODO:  UCF clean complaining about SSL Error
    #  requests.exceptions.SSLError: HTTPSConnectionPool(host='www.crcv.ucf.edu', port=443):
    #  Max retries exceeded with url: /data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip
    #  (Caused by SSLError(SSLCertVerificationError(1, '[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify
    #  failed: unable to get local issuer certificate (_ssl.c:1091)')))
    # "ucf101_clean": {
    #     "type": "source",
    #     "class_file": os.path.join(
    #         os.path.dirname(__file__), "build_classes", "ucf101_clean.py"
    #     ),
    # },
    "xview": {
        "type": "source",
        "class_file": os.path.join(
            os.path.dirname(__file__), "build_classes", "xview.py"
        ),
    },
    # TODO:  These are `Adversarial` Datasets from armory... need to
    #  determine if we need to do anything different here
    "apricot_dev": {
        "type": "source",
        "class_file": os.path.join(
            os.path.dirname(__file__), "build_classes", "adversarial", "apricot_dev.py"
        ),
    },
    "apricot_test": {
        "type": "source",
        "class_file": os.path.join(
            os.path.dirname(__file__), "build_classes", "adversarial", "apricot_test.py"
        ),
    },
    "carla_obj_det_dev": {
        "type": "source",
        "class_file": os.path.join(
            os.path.dirname(__file__),
            "build_classes",
            "adversarial",
            "carla_obj_det_dev.py",
        ),
    },
    "carla_obj_det_test": {
        "type": "source",
        "class_file": os.path.join(
            os.path.dirname(__file__),
            "build_classes",
            "adversarial",
            "carla_obj_det_test.py",
        ),
    },
    "carla_video_tracking_dev": {
        "type": "source",
        "class_file": os.path.join(
            os.path.dirname(__file__),
            "build_classes",
            "adversarial",
            "carla_video_tracking_dev.py",
        ),
    },
    "carla_video_tracking_test": {
        "type": "source",
        "class_file": os.path.join(
            os.path.dirname(__file__),
            "build_classes",
            "adversarial",
            "carla_video_tracking_test.py",
        ),
    },
    # TODO: dapricot builds fine but complains on construction about "ragged_flat_values" slice metod
    #  need to figure out what is going on there...for now commenting out
    #  TypeError: Only integers, slices (`:`), ellipsis (`...`), tf.newaxis (`None`) and scalar
    #  tf.int32/tf.int64 tensors are valid indices, got 'ragged_flat_values'
    "dapricot_dev": {
        "type": "source",
        "class_file": os.path.join(
            os.path.dirname(__file__), "build_classes", "adversarial", "dapricot_dev.py"
        ),
    },
    "dapricot_test": {
        "type": "source",
        "class_file": os.path.join(
            os.path.dirname(__file__),
            "build_classes",
            "adversarial",
            "dapricot_test.py",
        ),
    },
    "gtsrb_bh_poison_micronnet": {
        "type": "source",
        "class_file": os.path.join(
            os.path.dirname(__file__),
            "build_classes",
            "adversarial",
            "gtsrb_bh_poison_micronnet.py",
        ),
    },
    "imagenet_adversarial": {
        "type": "source",
        "class_file": os.path.join(
            os.path.dirname(__file__),
            "build_classes",
            "adversarial",
            "imagenet_adversarial.py",
        ),
    },
    "librispeech_adversarial": {
        "type": "source",
        "class_file": os.path.join(
            os.path.dirname(__file__),
            "build_classes",
            "adversarial",
            "librispeech_adversarial.py",
        ),
    },
    # TODO: Note the change where `_` is removed between adversarial and 224x224
    #  ask if this will be an issue
    "resisc45_densenet121_univpatch_and_univperturbation_adversarial224x224": {
        "type": "source",
        "class_file": os.path.join(
            os.path.dirname(__file__),
            "build_classes",
            "adversarial",
            "resisc45_densenet121_univpatch_and_univperturbation_adversarial224x224.py",
        ),
    },
    # TODO: Note the change where `_` is removed between adversarial and 224x224
    #  ask if this will be an issue...otherwise this works
    "ucf101_mars_perturbation_and_patch_adversarial112x112": {
        "type": "source",
        "class_file": os.path.join(
            os.path.dirname(__file__),
            "build_classes",
            "adversarial",
            "ucf101_mars_perturbation_and_patch_adversarial112x112.py",
        ),
    },
}


def build_tfds_dataset(dataset_name: str, local_path: str, feature_dict=None) -> str:
    log.info(f"Building Dataset: {dataset_name} from TFDS artifacts")
    log.debug("Constructing Builder Object")
    builder = tfds.builder(dataset_name, data_dir=local_path)
    log.debug("Downloading artifacts...")
    builder.download_and_prepare()
    ds_path = tools.get_ds_path(dataset_name, local_path)
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
        ds_path = tools.get_ds_path(dataset_name, local_path)
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
    log.info(f"Building Dataset: {dataset_name}")
    log.debug(f"\t using Config: {dataset_config}")
    ds_path = os.path.join(
        local_path, dataset_name
    )  # Not sure it exists...don't use tools.get_ds_path
    log.trace(f"Initial ds_path: {ds_path}")
    if os.path.isdir(ds_path):
        log.info(f"Dataset: {dataset_name} already exists at {ds_path}")
        if not clean:
            log.warning("...skipping build. To overwrite use `--clean`")
            ds_path = tools.get_ds_path(
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
            f"Supported Dataset Type: {SUPPORTED_DATASETS[dataset_name]} is not implemented"
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
        choices=["all"] + list(SUPPORTED_DATASETS.keys()),
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
        default=os.path.expanduser(os.path.join("~", ".armory", "dataset_builds")),
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
    log.remove()
    log.add(sys.stdout, level=args.verbosity.upper())

    # Listing SUPPORTED_DATASETS
    if args.list:
        print("Supported Datasets:\n\t" + "\n\t".join(SUPPORTED_DATASETS.keys()))
        exit()

    if args.dataset is not None:
        args.dataset = list(itertools.chain(*args.dataset))  # Flatten list
        if "all" in args.dataset:
            dataset_dict = SUPPORTED_DATASETS
        else:
            dataset_dict = {k: SUPPORTED_DATASETS[k] for k in args.dataset}
    else:
        args.local_class_path = list(itertools.chain(*args.local_class_path))
        dataset_dict = {
            os.path.splitext(os.path.basename(k))[0]: {
                "type": "source",
                "class_file": k,
            }
            for k in args.local_class_path
        }

    if len(dataset_dict.keys()) == 0:
        log.error("Must provide at least 1 dataset/class_file")
        raise ValueError("Must provide at least 1 dataset/class_file")

    log.info(f"Attempting to Build Datasets: {dataset_dict.keys()}")
    for ds_name, ds_config in dataset_dict.items():
        ds_path = build(ds_name, ds_config, args.output_directory, args.clean)
        try:
            ds_info, ds = tools.load(ds_name, args.output_directory)
        except Exception as e:
            log.exception(f"Could not reconstruct dataset located at {ds_path}!!")
            log.exception(e)
            raise e

    log.success("\t ALL Builds Complete !!")
