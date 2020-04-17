"""
Utils for data processing

"""
import logging
import hashlib
import tarfile
import os
import shutil
import random
import string
import json

import boto3
from botocore import UNSIGNED
from botocore.client import Config
from botocore.exceptions import ClientError
import requests
from tqdm import tqdm

from armory import paths
from armory.data.progress_percentage import ProgressPercentage

logger = logging.getLogger(__name__)

requests.packages.urllib3.disable_warnings(
    requests.packages.urllib3.exceptions.InsecureRequestWarning
)


def maybe_download_weights_from_s3(weights_file: str) -> str:
    """

    :param weights_file:
    :return:
    """
    saved_model_dir = paths.docker().saved_model_dir
    filepath = os.path.join(saved_model_dir, weights_file)

    if os.path.isfile(filepath):
        logger.info(f"Using available {weights_file} in Armory `saved_model_dir`")
    else:
        logger.info(
            f"{weights_file} not found in Armory `saved_model_dir`. Attempting to pull weights from S3"
        )
        download_file_from_s3(
            "armory-public-data",
            f"model-weights/{weights_file}",
            f"{saved_model_dir}/{weights_file}",
        )
    return filepath


def download_file_from_s3(bucket_name: str, key: str, local_path: str) -> None:
    """
    Downloads file from S3 anonymously
    :param bucket_name: S3 Bucket name
    :param key: S3 File key name
    :param local_path: Local file path to download as
    """
    if not os.path.isfile(local_path):
        client = boto3.client(
            "s3", config=Config(signature_version=UNSIGNED), verify=False
        )

        try:
            logger.info("Downloading S3 data file...")
            with ProgressPercentage(client, bucket_name, key) as Callback:
                client.download_file(bucket_name, key, local_path, Callback=Callback)
        except ClientError:
            raise KeyError(f"File {key} not available in {bucket_name} bucket.")

    else:
        logger.info(f"Reusing cached file {local_path}...")


def download_requests(url: str, dirpath: str, filename: str):
    filepath = os.path.join(dirpath, filename)
    chunk_size = 4096
    r = requests.get(url, stream=True, verify=False)
    with open(filepath, "wb") as f:
        progress_bar = tqdm(
            unit="B", total=int(r.headers["Content-Length"]), unit_scale=True
        )
        for chunk in r.iter_content(chunk_size=chunk_size):
            if chunk:  # filter keep-alive chunks
                progress_bar.update(len(chunk))
                f.write(chunk)


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


def move_merge(source, dest):
    dest_location = os.path.join(dest, os.path.basename(source))
    if not os.path.exists(dest_location):
        shutil.move(source, dest)
        return

        # Move up a directory, as shutil.move will error
    filepaths = [
        os.path.join(source, x) for x in os.listdir(source) if not x.startswith(".")
    ]
    if len(filepaths) != 1 or not os.path.isdir(filepaths[0]):
        raise ValueError(f"{source} not a single branch directory. Cannot recurse.")
    move_merge(filepaths[0], dest_location)
    os.rmdir(source)


def download_verify_dataset_cache(dataset_dir, checksum_file, name):
    with open(checksum_file, "r") as fh:
        s3_bucket_name, s3_key, file_length, hash = fh.readline().strip().split()

    # download
    cache_dir = os.path.join(dataset_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    tar_filepath = os.path.join(cache_dir, os.path.basename(s3_key))
    already_verified = False
    if os.path.exists(tar_filepath):
        # Check existing download to avoid falling back to processing data
        logger.info(f"{tar_filepath} exists. Verifying...")
        try:
            verify_size(tar_filepath, int(file_length))
            verify_sha256(tar_filepath, hash)
            already_verified = True
        except ValueError as e:
            logger.warning(f"Verification failed: {str(e)}")
            os.remove(tar_filepath)

    if not os.path.exists(tar_filepath):
        logger.info(f"Downloading dataset: {name}...")
        try:
            s3_url_region = "us-east-2"
            url = f"https://{s3_bucket_name}.s3.{s3_url_region}.amazonaws.com/{s3_key}"
            download_requests(url, dataset_dir, tar_filepath)
        except KeyboardInterrupt:
            logger.exception("Keyboard interrupt caught")
            if os.path.exists(tar_filepath):
                os.remove(tar_filepath)
            raise
    else:
        logger.info("Dataset already downloaded.")

    # verification
    if not already_verified:
        try:
            verify_size(tar_filepath, int(file_length))
            logger.info("Verifying sha256 hash of download...")
            verify_sha256(tar_filepath, hash)
        except ValueError:
            if os.path.exists(tar_filepath):
                os.remove(tar_filepath)
            logger.warning(
                "Cached file download failed. Falling back to processing data..."
            )
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
        logger.warning(f"Could not read tarfile: {tar_filepath}")
        logger.warning("Falling back to processing data...")
        return
    except tarfile.ExtractError:
        logger.warning(f"Could not extract tarfile: {tar_filepath}")
        logger.warning("Falling back to processing data...")
        return

    filepaths = [
        os.path.join(tmp_dir, x) for x in os.listdir(tmp_dir) if not x.startswith(".")
    ]
    if len(filepaths) != 1 or not os.path.isdir(filepaths[0]):
        raise ValueError(
            f"{tmp_dir} not a single branch directory. tfrecord archive corrupted."
        )
    move_merge(filepaths[0], dataset_dir)
    try:
        shutil.rmtree(tmp_dir)
    except OSError as e:
        if not isinstance(e, FileNotFoundError):
            logger.exception(f"Error removing temporary directory {tmp_dir}")


def _read_validate_scenario_config(config_filepath):
    with open(config_filepath) as f:
        config = json.load(f)
    if "scenario" not in config.keys():
        raise ValueError("Does not match config schema")
    if not isinstance(config["scenario"], dict):
        raise ValueError('config["scenario"] must be dictionary')
    return config
