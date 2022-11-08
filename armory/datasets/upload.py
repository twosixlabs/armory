import os

import boto3

from armory.datasets import common, progress
from armory.logs import log


def upload_file_to_s3(key: str, local_path: str, public: bool = False):
    """
    Uploads a file to S3 using credentials stored in ENV variables.

    key - S3 File keyname
    local_path - Local file path to upload
    public - if true, upload to armory public, otherwise upload to armory private
    """
    client = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("ARMORY_PRIVATE_S3_ID"),
        aws_secret_access_key=os.getenv("ARMORY_PRIVATE_S3_KEY"),
    )
    log.info("Uploading file to S3...")
    bucket = common.get_bucket(public=public)
    ExtraArgs = {"ACL": "public-read"} if public else None
    client.upload_file(
        local_path,
        bucket,
        key,
        Callback=progress.ProgressPercentageUpload(local_path),
        ExtraArgs=ExtraArgs,
    )
    print()  # Callback progress does not add a final "\n"


def upload(name, public=False):
    try:
        info = common.cached_datasets()[name]
        version = info["version"]
    except KeyError:
        raise KeyError("Must call `package.update(filepath)` before uploading")

    filepath = common.get_cache_dataset_path(name, version)
    if not filepath.is_file():
        raise FileNotFoundError(f"Cache file {filepath} not found.")

    if not public:
        raise ValueError("Uploading to private s3 not currently supported")

    key = common.get_cache_key(name, version)
    upload_file_to_s3(str(key), local_path=str(filepath), public=public)
