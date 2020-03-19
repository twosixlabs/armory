"""
Utils for data processing

"""
import logging
import hashlib
import tarfile
import os
import subprocess
import shutil
import random
import string

import boto3
from botocore import UNSIGNED
from botocore.client import Config


logger = logging.getLogger(__name__)


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


def sha256(filepath: str, block_size=4096):
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(block_size), b""):
            sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()


def verify_sha256(filepath: str, hash_value: str, block_size: int = 4096):
    """
    Verify that the target filepath has the given sha256 hash_value
        Raise ValueError if False

    filepath - target filepath
    hash_value - hex encoded value of the hash
    block_size - block size for chunked reading from file
    """

    if len(hash_value) != 64:
        raise ValueError(f"Invalid hash_value: len({hash_value}) != 64")
    hash_value = hash_value.lower()
    if not all(x in "0123456789abcdef" for x in hash_value):
        raise ValueError(f"Invalid hash_value: {hash_value} contains non-hex chars")

    value = sha256(filepath, block_size=block_size)
    if value != hash_value:
        raise ValueError(f"sha256 hash of {filepath}: {value} != {hash_value}")


def verify_size(filepath: str, file_size: int):
    size = os.path.getsize(filepath)
    if size != file_size:
        raise ValueError(f"file size of {filepath}: {size} != {file_size}")


def download_verify_dataset_cache(dataset_dir, checksum_file):
    with open(checksum_file, "r") as fh:
        url, file_length, hash = fh.readline().strip().split()
    # download
    cache_dir = os.path.join(dataset_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    tar_filepath = os.path.join(cache_dir, os.path.basename(url))
    if not os.path.exists(tar_filepath):
        curl(url, dataset_dir, tar_filepath)

    # verification
    try:
        verify_size(tar_filepath, int(file_length))
        logger.info("Verifying sha256 hash of download...")
        verify_sha256(tar_filepath, hash)
    except ValueError:
        os.remove(tar_filepath)
        logger.info("Cached file download failed. Falling back to processing data...")
        return

    tmp_dir = os.path.join(
        cache_dir,
        "tmp_" + "".join(random.choice(string.ascii_lowercase) for _ in range(16)),
    )
    os.makedirs(tmp_dir)

    logger.info("Extracting .tfrecord files from download...")
    try:
        with tarfile.open(tar_filepath, "r:gz") as tar_ref:
            tar_ref.extractall(tmp_dir)
    except tarfile.ReadError:
        logger.info(f"Could not read tarfile: {tar_filepath}")
        logger.info("Falling back to processing data...")
        return
    except tarfile.ExtractError:
        logger.info(f"Could not extract tarfile: {tar_filepath}")
        logger.info("Falling back to processing data...")
        return

    # move tmp_dir subdirectory to dataset_dir (tarball does not contain files at root level)
    for directory in os.listdir(path=tmp_dir):
        shutil.move(os.path.join(tmp_dir, directory), dataset_dir)
    os.rmdir(tmp_dir)
