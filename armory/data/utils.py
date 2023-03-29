"""
Utils for data processing

"""
import hashlib
import json
import os
import random
import shutil
import string
import subprocess
import tarfile

import boto3
from botocore import UNSIGNED
from botocore.client import Config
from botocore.exceptions import ClientError
import requests
from tqdm import tqdm

from armory import paths
from armory.configuration import get_verify_ssl
from armory.data.progress_percentage import ProgressPercentage, ProgressPercentageUpload
from armory.logs import is_progress, log

CHECKSUMS_DIRS = []


def add_checksums_dir(dir):
    global CHECKSUMS_DIRS
    CHECKSUMS_DIRS.append(dir)


def maybe_download_weights_from_s3(
    weights_file: str, *, auto_expand_tars: bool = False
) -> str:
    """

    :param weights_file:
    :param auto_expand_tars:
    :return:
    """
    saved_model_dir = paths.runtime_paths().saved_model_dir
    filepath = os.path.join(saved_model_dir, weights_file)

    if os.path.isfile(filepath):
        log.info(f"Using available {weights_file} in Armory `saved_model_dir`")
    else:
        log.info(
            f"{weights_file} not found in Armory `saved_model_dir`. Attempting to pull weights from S3"
        )
        try:
            download_file_from_s3(
                "armory-public-data",
                f"model-weights/{weights_file}",
                f"{saved_model_dir}/{weights_file}",
            )
        except KeyError:
            if (
                "ARMORY_INCLUDE_SUBMISSION_BUCKETS" in os.environ
                and os.getenv("ARMORY_INCLUDE_SUBMISSION_BUCKETS") != ""
            ):
                try:
                    download_private_file_from_s3(
                        "armory-submission-data",
                        f"model-weights/{weights_file}",
                        f"{saved_model_dir}/{weights_file}",
                    )

                except KeyError:
                    raise ValueError(
                        (
                            f"{weights_file} was not found in the armory public & submission S3 buckets."
                        )
                    )
            else:
                raise ValueError(
                    (
                        f"{weights_file} was not found in the armory S3 bucket. If "
                        "you're attempting to load a custom set of weights for "
                        "your model be sure that they are available in the armory "
                        "`saved_model_dir` directory on your host environment."
                    )
                )

    if auto_expand_tars:
        if tarfile.is_tarfile(filepath):
            log.debug(f"Detected model weights file {weights_file} as a tar archive")
            with tarfile.open(filepath) as tar:
                # check if the tarfile contains a directory containing all its members
                # ie if the tarfile expands out entirely into a subdirectory
                dirs = [fi.name for fi in tar.getmembers() if fi.isdir()]
                commonpath = os.path.commonpath(tar.getnames())
                if not commonpath or commonpath not in dirs:
                    raise PermissionError(
                        (
                            f"{weights_file} does not expand into a subdirectory."
                            f" Weights files submitted as tarballs must expand into a subdirectory."
                        )
                    )
                full_path = os.path.join(saved_model_dir, commonpath)
                if os.path.exists(full_path):
                    log.warning(
                        f"Model weights folder {commonpath} from {weights_file} already exists"
                    )
                    log.warning(f"Skipping auto-unpacking of {weights_file}")
                    log.warning(f"Delete {commonpath} manually to force unpacking")
                else:
                    log.info(f"Auto-unpacking model weights from {weights_file}")
                    tar.extractall(path=saved_model_dir)
            filepath = commonpath

    return filepath


def download_file_from_s3(bucket_name: str, key: str, local_path: str) -> None:
    """
    Downloads file from S3 anonymously
    :param bucket_name: S3 Bucket name
    :param key: S3 File key name
    :param local_path: Local file path to download as
    """
    verify_ssl = get_verify_ssl()
    if not os.path.isfile(local_path):
        client = boto3.client(
            "s3", config=Config(signature_version=UNSIGNED), verify=verify_ssl
        )

        try:
            log.info(f"downloading S3 data file {bucket_name}/{key}")
            total = client.head_object(Bucket=bucket_name, Key=key)["ContentLength"]
            if is_progress():
                with ProgressPercentage(client, bucket_name, key, total) as Callback:
                    client.download_file(
                        bucket_name, key, local_path, Callback=Callback
                    )
            else:
                client.download_file(bucket_name, key, local_path)
        except ClientError:
            raise KeyError(f"File {key} not available in {bucket_name} bucket.")

    else:
        log.info(f"Reusing cached file {local_path}...")


def download_private_file_from_s3(bucket_name: str, key: str, local_path: str):
    """
    Downloads file from S3 using credentials stored in ENV variables.
    :param bucket_name: S3 Bucket name
    :param key: S3 File keyname
    :param local_path: Local file path to download as
    """
    verify_ssl = get_verify_ssl()
    if not os.path.isfile(local_path):
        client = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("ARMORY_PRIVATE_S3_ID"),
            aws_secret_access_key=os.getenv("ARMORY_PRIVATE_S3_KEY"),
            verify=verify_ssl,
        )
        try:
            log.info(f"downloading S3 data file {bucket_name}/{key}")
            total = client.head_object(Bucket=bucket_name, Key=key)["ContentLength"]
            if is_progress():
                with ProgressPercentage(client, bucket_name, key, total) as Callback:
                    client.download_file(
                        bucket_name, key, local_path, Callback=Callback
                    )
            else:
                client.download_file(bucket_name, key, local_path)
        except ClientError:
            raise KeyError(f"File {key} not available in {bucket_name} bucket.")
    else:
        log.info("Reusing cached S3 data file...")


def download_requests(url: str, dirpath: str, filename: str):
    verify_ssl = get_verify_ssl()

    filepath = os.path.join(dirpath, filename)
    chunk_size = 4096
    r = requests.get(url, stream=True, verify=verify_ssl)
    with open(filepath, "wb") as f:
        progress_bar = None
        if is_progress():
            progress_bar = tqdm(
                unit="B", total=int(r.headers["Content-Length"]), unit_scale=True
            )
        for chunk in r.iter_content(chunk_size=chunk_size):
            if chunk:  # filter keep-alive chunks
                if progress_bar:
                    progress_bar.update(len(chunk))
                f.write(chunk)

    log.info(f"downloaded {filename} from {url}")


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
    found_checksum_flag = False
    log.info("Attempting download_verigy_dataset_cache with dataset_dir")
    for checksum_dir in CHECKSUMS_DIRS:
        checksum_file_full_path = os.path.join(checksum_dir, checksum_file)
        if os.path.exists(checksum_file_full_path):
            found_checksum_flag = True
            break
    if not found_checksum_flag:
        raise FileNotFoundError(f"Could not locate checksum file {checksum_file}")

    with open(checksum_file_full_path, "r") as fh:
        s3_bucket_name, s3_key, file_length, hash = fh.readline().strip().split()

    # download
    cache_dir = os.path.join(dataset_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    tar_filepath = os.path.join(cache_dir, os.path.basename(s3_key))
    already_verified = False
    if os.path.exists(tar_filepath):
        # Check existing download to avoid falling back to processing data
        log.info(f"{tar_filepath} exists. Verifying...")
        try:
            verify_size(tar_filepath, int(file_length))
            verify_sha256(tar_filepath, hash)
            already_verified = True
        except ValueError as e:
            log.warning(f"Verification failed: {str(e)}")
            os.remove(tar_filepath)

    if not os.path.exists(tar_filepath):
        if s3_bucket_name == "local":
            raise FileNotFoundError(f"Expected to find {s3_key} locally in cache!")
        log.info(f"Downloading dataset: {name}...")
        try:
            s3_url_region = "us-east-2"
            url = f"https://{s3_bucket_name}.s3.{s3_url_region}.amazonaws.com/{s3_key}"
            download_requests(url, dataset_dir, tar_filepath)
        except KeyboardInterrupt:
            log.exception("Keyboard interrupt caught")
            if os.path.exists(tar_filepath):
                os.remove(tar_filepath)
            raise
    else:
        log.info("Dataset already downloaded.")

    # verification
    if not already_verified:
        try:
            verify_size(tar_filepath, int(file_length))
            log.info("Verifying sha256 hash of download...")
            verify_sha256(tar_filepath, hash)
        except ValueError:
            if os.path.exists(tar_filepath):
                os.remove(tar_filepath)
            log.warning(
                "Cached file download failed. Falling back to processing data..."
            )
            return

    tmp_dir = os.path.join(
        cache_dir,
        "tmp_" + "".join(random.choice(string.ascii_lowercase) for _ in range(16)),
    )
    os.makedirs(tmp_dir)

    log.info("Extracting .tfrecord files from download...")
    try:
        completedprocess = subprocess.run(
            ["tar", "zxvf", tar_filepath, "--directory", tmp_dir],
        )
        if completedprocess.returncode:
            log.warning("bash tar failed. Reverting to python tar unpacking")
            with tarfile.open(tar_filepath, "r:gz") as tar_ref:
                tar_ref.extractall(tmp_dir)
    except tarfile.ReadError:
        log.warning(f"Could not read tarfile: {tar_filepath}")
        log.warning("Falling back to processing data...")
        return
    except tarfile.ExtractError:
        log.warning(f"Could not extract tarfile: {tar_filepath}")
        log.warning("Falling back to processing data...")
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
            log.exception(f"Error removing temporary directory {tmp_dir}")


def _read_validate_scenario_config(config_filepath):
    with open(config_filepath) as f:
        config = json.load(f)
    if "scenario" not in config.keys():
        raise ValueError("Does not match config schema")
    if not isinstance(config["scenario"], dict):
        raise ValueError('config["scenario"] must be dictionary')
    return config


def upload_file_to_s3(key: str, local_path: str, public: bool = False):
    """
    Uploads a file to S3 using credentials stored in ENV variables.
    :param key: S3 File keyname
    :param local_path: Local file path to download as
    :param public: boolean to choose private or public bucket
    """
    client = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("ARMORY_PRIVATE_S3_ID"),
        aws_secret_access_key=os.getenv("ARMORY_PRIVATE_S3_KEY"),
    )
    log.info("Uploading file to S3...")
    if public:
        client.upload_file(
            local_path,
            "armory-public-data",
            key,
            Callback=ProgressPercentageUpload(local_path),
            ExtraArgs={"ACL": "public-read"},
        )

    else:
        client.upload_file(local_path, "armory-private-data", key)
