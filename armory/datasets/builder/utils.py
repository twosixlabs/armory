import glob
import importlib
import importlib.util
import os
import re
import sys

from loguru import logger as log
import tensorflow_datasets as tfds

print(f"User: {os.path.expanduser('~')}")
DEFAULT_DATASET_DIRECTORY = os.path.expanduser("~/.armory/dataset_builds/")
if not os.path.isdir(DEFAULT_DATASET_DIRECTORY):
    log.warning(
        f"Dataset Build Directory: {DEFAULT_DATASET_DIRECTORY} does not exist...creating!!"
    )
    os.makedirs(DEFAULT_DATASET_DIRECTORY)

BUILD_CLASS_DIR = os.path.join(os.path.dirname(__file__), "build_classes")


def camel_case(s: str, capitalize_first: bool = False) -> str:
    """Returns Camel Case Version of String
    Parameters:
        s (str):                        The string to convert (e.g. test_my_name)
        capitalize_first (bool):        Make first letter uppercase (e.g. for Class Name)

    Returns:
        camel_s (str):                  The converted CamelCase string

    """
    s = re.sub(r"(_|-)+", " ", s).split(" ")
    s = "".join([i[0].upper() + i[1:] for i in s])
    if not capitalize_first:
        s = s[0].lower() + s[1:]
    return s


def get_local_config(python_class_file: str):
    """Return a Config for a locally constructed TFDS dataset
    Parameters:
        python_class_file (str):        Path to the TFDS Generator class definition file

    Returns:
        config (dict):                  The necessary config dictionary for building datasets
    """
    if not os.path.isfile(python_class_file):
        raise ValueError(f"TFDS Class File: {python_class_file} does not exist!!")

    fname, fext = os.path.splitext(os.path.basename(python_class_file))
    if fext != ".py":
        raise ValueError(
            f"TFDS Class File: {python_class_file} must be a python file with extension `.py`"
        )

    try:
        spec = importlib.util.spec_from_file_location(
            "build_classes", python_class_file
        )
        foo = importlib.util.module_from_spec(spec)
        sys.path.append(os.path.dirname(python_class_file))
        with tfds.core.registered.skip_registration():
            spec.loader.exec_module(foo)
            clsname = camel_case(fname, capitalize_first=True)
            cls = getattr(foo, clsname)
            config = {
                "type": "source",
                "class_file": python_class_file,
                "expected_name": clsname,
                "expected_version": cls.VERSION,
            }
        return config
    except Exception as e:
        log.error(f"Could Not parse local TFDS class file: {python_class_file}!!")
        raise e


# TODO:  Resolve Dataset Issues Listed below
#  - ucf_clean complaining about SSL Error
#    requests.exceptions.SSLError: HTTPSConnectionPool(host='www.crcv.ucf.edu', port=443):
#    Max retries exceeded with url: /data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip
#    (Caused by SSLError(SSLCertVerificationError(1, '[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify
#    failed: unable to get local issuer certificate (_ssl.c:1091)')))
#  - dapricot builds fine but complains on construction about "ragged_flat_values" slice method
#    TypeError: Only integers, slices (`:`), ellipsis (`...`), tf.newaxis (`None`) and scalar
#    tf.int32/tf.int64 tensors are valid indices, got 'ragged_flat_values'
#  - coco/2017 was not building locally (too big) need to try on larger machine
#

SUPPORTED_DATASETS = {
    #    "mnist": {
    #        "type": "tfds",
    #        "feature_dict": {
    #            "type": "tensorflow_datasets.core.features.features_dict.FeaturesDict",
    #            "content": json.dumps(
    #                {
    #                    "features": {
    #                        "image": {
    #                            "pythonClassName": "tensorflow_datasets.core.features.image_feature.Image",
    #                            "image": {
    #                                "shape": {"dimensions": ["-1", "-1", "1"]},
    #                                "dtype": "uint8",
    #                            },
    #                        },
    #                        "label": {
    #                            "pythonClassName": "tensorflow_datasets.core.features.class_label_feature.ClassLabel",
    #                            "classLabel": {"numClasses": "10"},
    #                        },
    #                    }
    #                }
    #            ),
    #            "proto_cls": "tensorflow_datasets.FeaturesDict",
    #        },
    #        "expected_name": "mnist",
    #        "expected_version": "3.0.1",
    #    },
    #    "cifar10": {
    #        "type": "tfds",
    #        "feature_dict": None,
    #        "expected_name": "cifar10",
    #        "expected_version": "3.0.2",
    #    },
    #    "cifar100": {
    #        "type": "tfds",
    #        "feature_dict": None,
    #        "expected_name": "cifar100",
    #        "expected_version": "3.0.2",
    #    },
    #    "imagenette/full-size": {
    #        "type": "tfds",
    #        "feature_dict": None,
    #        "expected_name": "imagenette/full-size",
    #        "expected_version": "1.0.0",
    #    },
    #    # "coco/2017": {"type": "tfds", "feature_dict": None, "expected_name": "coco/2017", "expected_version": "1.0.0"},
    #    "so2sat/all": {
    #        "type": "tfds",
    #        "feature_dict": None,
    #        "expected_name": "so2sat/all",
    #        "expected_version": "2.1.0",
    #    },
    #    "carla_obj_det_train": get_local_config(
    #        os.path.join(BUILD_CLASS_DIR, "carla_obj_det_train.py")
    #    ),
    "digit": get_local_config(os.path.join(BUILD_CLASS_DIR, "digit.py")),
    #    "german_traffic_sign": get_local_config(
    #        os.path.join(BUILD_CLASS_DIR, "german_traffic_sign.py")
    #    ),
    #    "librispeech_full": get_local_config(
    #        os.path.join(BUILD_CLASS_DIR, "librispeech_full.py")
    #    ),
    #    "librispeech_dev_clean_split": get_local_config(
    #        os.path.join(BUILD_CLASS_DIR, "librispeech_dev_clean_split.py")
    #    ),
    #    "resisc10_poison": get_local_config(
    #        os.path.join(BUILD_CLASS_DIR, "resisc10_poison.py")
    #    ),
    #    "resisc45_split": get_local_config(
    #        os.path.join(BUILD_CLASS_DIR, "resisc45_split.py")
    #    ),
    #    # "ucf101_clean": get_local_config(os.path.join(BUILD_CLASS_DIR, "ucf101_clean.py")), # See TODO Above...SSL Error
    #    "xview": get_local_config(os.path.join(BUILD_CLASS_DIR, "xview.py")),
    #    # Below Here are Adversarial Datasets
    #    "apricot_dev": get_local_config(
    #        os.path.join(BUILD_CLASS_DIR, "adversarial", "apricot_dev.py")
    #    ),
    #    "apricot_test": get_local_config(
    #        os.path.join(BUILD_CLASS_DIR, "adversarial", "apricot_test.py")
    #    ),
    #    "carla_obj_det_dev": get_local_config(
    #        os.path.join(BUILD_CLASS_DIR, "adversarial", "carla_obj_det_dev.py")
    #    ),
    #    "carla_obj_det_test": get_local_config(
    #        os.path.join(BUILD_CLASS_DIR, "adversarial", "carla_obj_det_test.py")
    #    ),
    #    "carla_video_tracking_dev": get_local_config(
    #        os.path.join(BUILD_CLASS_DIR, "adversarial", "carla_video_tracking_dev.py")
    #    ),
    #    "carla_video_tracking_test": get_local_config(
    #        os.path.join(BUILD_CLASS_DIR, "adversarial", "carla_video_tracking_test.py")
    #    ),
    #    # "dapricot_dev": get_local_config(os.path.join(BUILD_CLASS_DIR, "adversarial", "dapricot_dev.py")),  # See TODO above about error
    #    "dapricot_test": get_local_config(
    #        os.path.join(BUILD_CLASS_DIR, "adversarial", "dapricot_test.py")
    #    ),
    #    "gtsrb_bh_poison_micronnet": get_local_config(
    #        os.path.join(BUILD_CLASS_DIR, "adversarial", "gtsrb_bh_poison_micronnet.py")
    #    ),
    #    "imagenet_adversarial": get_local_config(
    #        os.path.join(BUILD_CLASS_DIR, "adversarial", "imagenet_adversarial.py")
    #    ),
    #    "librispeech_adversarial": get_local_config(
    #        os.path.join(BUILD_CLASS_DIR, "adversarial", "librispeech_adversarial.py")
    #    ),
    #    "resisc45_densenet121_univpatch_and_univperturbation_adversarial224x224": get_local_config(
    #        os.path.join(
    #            BUILD_CLASS_DIR,
    #            "adversarial",
    #            "resisc45_densenet121_univpatch_and_univperturbation_adversarial224x224.py",
    #        )
    #    ),
    #    "ucf101_mars_perturbation_and_patch_adversarial112x112": get_local_config(
    #        os.path.join(
    #            BUILD_CLASS_DIR,
    #            "adversarial",
    #            "ucf101_mars_perturbation_and_patch_adversarial112x112.py",
    #        )
    #    ),
}


def setup_logger(level="info", suppress_tfds_progress=False):
    """Setup the Loguru Logger to the appropriate level for stdout"""
    log.remove()
    log.add(sys.stdout, level=level)
    if suppress_tfds_progress:
        tfds.disable_progress_bar()


def validate_dataset_directory_contents(dataset_full_path: str):
    """Checks Dataset Directory to ensure proper structure
    This function will check the provided path to make sure it exists
    and conforms to the TFDS dataset structure format.

    Parameters:
        dataset_full_path (str):            Full Path to the Constructed Dataset Directory

    Returns:
        null:                               Note: This will raise errors, if encountered
                                            so if it returns, the directory is valid
    """

    if not os.path.isdir(dataset_full_path):
        msg = f"Dataset: {dataset_full_path} does not exist or is not a directory!!"
        log.error(msg)
        raise ValueError(msg)

    tfrecords = glob.glob(os.path.join(dataset_full_path, "*.tfrecord*"))
    meta = [
        os.path.basename(i)
        for i in glob.glob(os.path.join(dataset_full_path, "*.json"))
    ]
    if (
        len(tfrecords) == 0
        or "features.json" not in meta
        or "dataset_info.json" not in meta
    ):
        msg = f"Dataset: {dataset_full_path} does does not contain appropriate files...checking to see if this is a parent!!"
        log.warning(msg)
        versions = next(os.walk(dataset_full_path))[1]
        if len(versions) != 1:
            msg = f"Dataset: {dataset_full_path} does not contain appropriate files and does not have correct version subdir!!"
            log.error(msg)
            raise ValueError(msg)
        else:
            pth = validate_dataset_directory_contents(
                os.path.join(dataset_full_path, versions[0])
            )
            return pth
    return dataset_full_path


def get_dataset_full_path(
    dataset_name: str,
    dataset_directory: str = DEFAULT_DATASET_DIRECTORY,
    validate: bool = True,
) -> str:
    """Returns the Full Path to the locally constructed dataset
    Parameters:
        dataset_name (str):         Name of the Dataset/Directory (e.g. `mnist`, `so2sat/all`, etc.)
                                     NOTE: This will attempt to resolve the version and expects exactly
                                     one version for a given dataset.
        dataset_directory (str):    Parent directory containing the Dataset
        validate (bool):            Validate the full_path to make sure it exists and has correct structure

    Returns:
        dataset_path (str):         This will be the full path to the dataset contents
                                    which will include the dataset version.  For example
                                    `mnist` would return [dataset_directory]/mnist/3.0.1/

    """
    ds_path = os.path.join(dataset_directory, dataset_name)
    if not os.path.isdir(ds_path):
        raise ValueError(
            f"Dataset: {dataset_directory}/{dataset_name} does not exist!!"
        )
    versions = next(os.walk(ds_path))[1]
    if len(versions) != 1:
        raise RuntimeError(
            f"Dataset: {dataset_directory}/{dataset_name} has len(versions) == {len(versions)} != 1!!"
        )

    ds_path = os.path.join(ds_path, versions[0])
    if validate:
        validate_dataset_directory_contents(ds_path)
        if (
            dataset_name in SUPPORTED_DATASETS
            and SUPPORTED_DATASETS[dataset_name]["expected_version"] != versions[0]
        ):
            raise RuntimeError(
                f"Dataset Path: {ds_path} resolved version: {versions[0]} "
                f"and is different from SUPPORTED_DATASETS Version: "
                f"{SUPPORTED_DATASETS[dataset_name]['expected_version']}!!!"
            )
    return ds_path


def get_dataset_archive_name(dataset_name: str, dataset_version: str) -> str:
    """Returns the archive file (.tar.gz) name given the dataset information
    Parameters:
        dataset_name (str):         Name of the Dataset/Directory (e.g. `mnist`, `so2sat/all`, etc.)
        dataset_version (str):      Version Number of the dataset

    Returns:
        archive_name (str):         Name used to make the archive file (e.g. mnist_3.0.0.tar.gz)

    """
    fname = "_".join(dataset_name.split("/")) + f"_{dataset_version}.tar.gz"
    return fname


def resolve_dataset_directories(
    datasets: list = [],
    parents: list = [],
    dataset_directory: str = DEFAULT_DATASET_DIRECTORY,
) -> list:
    """Returns a list of full_paths to dataset directories, given the inputs
    Parameters:
        datasets (list):                    List of Dataset Names or Paths (e.g. `mnist` or [data_dir]/mnist/3.0.2)
        parents (list):                     List of Parent Directories that contain datasets
        dataset_directory (str):            Used when `datasets` element is a `SUPPORTED_DATASETS` name

    Returns:
        dataset_directories (list):         List of Fully resolved, validated dataset paths
    """

    log.debug(f"Requested Dataset Directories: {datasets}")
    data_dirs = []
    for d in datasets:
        if d in SUPPORTED_DATASETS.keys():
            tmp = get_dataset_full_path(d, dataset_directory)
        elif os.path.isdir(os.path.join(dataset_directory, d)):
            tmp = os.path.join(dataset_directory, d)
        elif os.path.isdir(d):
            tmp = d
        else:
            raise ValueError(
                f"Cannot resolve Dataset: {d}...must be one of "
                f"[ `SUPPORTED_DATASETS` key | "
                f"sub-directory of `datset_directory` | "
                f"local_path to directory ]"
            )

        ds_path = validate_dataset_directory_contents(tmp)
        data_dirs.append(ds_path)

    log.debug(f"Requested Parents: {parents}")

    for d in parents:
        if not os.path.isdir(d):
            raise ValueError(
                f" Cannot resolve Parent Directory: {d} because it is not a directory!!"
            )
        for subd in next(os.walk(d))[1]:
            ds_path = validate_dataset_directory_contents(os.path.join(d, subd))
            data_dirs.append(ds_path)

    log.debug(f"Resolved Data Directories: {data_dirs}")

    if len(data_dirs) == 0:
        log.error("Must provide at least one Dataset!!")
        exit(1)

    return data_dirs
