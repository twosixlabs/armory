"""
Utils for data processing

"""
import os
import logging
import subprocess
from importlib import import_module

import boto3
from botocore import UNSIGNED
from botocore.client import Config


logger = logging.getLogger(__name__)


def load_dataset(dataset_config, *args, **kwargs):
    """
    Return dataset or raise KeyError

    Convenience function, essentially.
    """
    dataset_module = import_module(dataset_config["module"])
    dataset_fn = getattr(dataset_module, dataset_config["name"])
    return dataset_fn(*args, **kwargs)


def download_file_from_s3(bucket_name: str, key: str, local_path: str):
    """
    Downloads file from S3 anonymously
    :param bucket_name: S3 Bucket name
    :param key: S3 File keyname
    :param local_path: Local file path to download as
    """
    if not os.path.isfile(local_path):
        client = boto3.client("s3", config=Config(signature_version=UNSIGNED))
        logger.info("Downloading S3 data file...")
        client.download_file(bucket_name, key, local_path)
    else:
        logger.info("Reusing cached file...")


def curl(url: str, dirpath: str, filename: str) -> None:
    """
    Downloads a file with a specified output filename and directory
    :param url: URL to file
    :param dirpath: Output directory
    :param filename: Output filename
    """
    try:
        subprocess.check_call(["curl", "-L", url, "--output", filename], cwd=dirpath)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"curl command not found. Is curl installed? {e}")
    except subprocess.CalledProcessError:
        raise subprocess.CalledProcessError
