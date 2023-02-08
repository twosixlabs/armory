from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.client import Config
from botocore.exceptions import ClientError

from armory.logs import log
from armory.datasets import common, progress
from armory.configuration import get_verify_ssl


def download_file_from_s3(key: str, local_path: str, public: bool = True) -> None:
    """
    Downloads file from S3 anonymously

    key - S3 File key name
    local_path - local file path to download as
    public - whether to download from armory public (or private)
    """
    if not public:
        raise NotImplementedError("Does not yet work for armory private datasets")
    bucket = common.get_bucket(public=public)

    if Path(local_path).exists():
        raise FileExistsError(f"Cannot download. {local_path} already exists.")
    verify_ssl = get_verify_ssl()
    client = boto3.client(
        "s3", config=Config(signature_version=UNSIGNED), verify=verify_ssl
    )

    try:
        log.info(f"downloading S3 data file {bucket}/{key}")
        total = client.head_object(Bucket=bucket, Key=key)["ContentLength"]
        with progress.ProgressPercentage(client, bucket, key, total) as Callback:
            client.download_file(bucket, key, local_path, Callback=Callback)
    except ClientError:
        raise KeyError(f"File {key} not available in {bucket} bucket.")


def download(
    name,
    data_dir: str = None,
    public: bool = True,
    overwrite: bool = False,
    verify: bool = True,
):
    info = common.cached_datasets()[name]
    version = info["version"]
    filepath = common.get_cache_dataset_path(name, version, data_dir=data_dir)

    # verify filepath or delete it
    if filepath.is_file():
        if verify:
            try:
                common.verify_hash(filepath, info["size"], info["sha256"])
                log.info(
                    "Previous file exists and passes verification. Not downloading"
                )
                return filepath
            except ValueError:
                pass

        if overwrite:
            filepath.unlink()
        else:
            raise ValueError(
                f"set overwrite=True to overwrite cached dataset tarfile at {filepath}"
            )

    key = common.get_cache_key(name, version)
    download_file_from_s3(str(key), str(filepath), public=public)
    if not filepath.is_file():
        raise FileNotFoundError(f"Error downloading dataset {name} to {filepath}")

    # verify download
    if verify:
        common.verify_hash(filepath, info["size"], info["sha256"])
