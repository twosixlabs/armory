import os
import logging

import boto3
from armory.utils.progress_percentage import ProgressPercentage

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
    logger.info("Uploading file to S3...")
    if public:
        client.upload_file(
            local_path,
            "armory-public-data",
            key,
            Callback=ProgressPercentage(local_path),
            ExtraArgs={"ACL": "public-read"},
        )

    else:
        client.upload_file(local_path, "armory-private-data", key)
