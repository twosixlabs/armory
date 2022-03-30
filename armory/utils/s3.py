import boto3
import botocore.client
from botocore import UNSIGNED
from botocore.client import Config
from botocore.exceptions import ClientError
from armory.data.progress_percentage import ProgressPercentage, ProgressPercentageUpload
import os
from armory.logs import log

ANONYMOUS_CONFIG = Config(signature_version=UNSIGNED)


def download(
    bucket: str,
    key: str,
    local_path: str,
    config: botocore.client.Config = None,
    region_name: str = None,
    aws_access_key_id: str = os.getenv("ARMORY_PRIVATE_S3_ID", default=None),
    aws_secret_access_key: str = os.getenv("ARMORY_PRIVATE_S3_KEY", default=None),
    verify_ssl: bool = False,
    use_cache: bool = False,
    show_progress: bool = True,
) -> str:
    """Download File from armory S3
    Parameters:
        bucket (str):                       The name of the s3 bucket to use
        key (str):                          The name of file in the bucket to download
        local_path (str):                   Where to put the downloaded file
        config (botocore.client.Config):    Boto Client Config.  For example if you want to download anonymously
                                             you can use `config=armory.utils.s3.ANONYMOUS_CONFIG`
                                             (default: None)
        region_name (str):                  AWS region for the client to use.
        aws_access_key_id (str):            AWS key ID for S3 access.  If `config=None` this should be set!
                                                (default: None)
        aws_secret_access_key (str):        AWS Secret key for S3 access.  If `config=None` this should be set!
                                                (default: None)
        verify_ssl (bool):                  Whether or not the client should verify SSL certificates
        use_cache (bool):                   Use file (if exists) at `local_path`. (default: False)
        show_progress (bool):               Whether or not to use armory.logs.ProgressPercentage to show progress bar
                                            during download.

    Returns:
            local_path (str):               The path to the resulting downloaded file.


    """

    if use_cache and os.path.isfile(local_path):
        log.info(
            f"S3 File already exists locally at {local_path}... skipping download!"
        )
        return local_path
    else:
        log.debug("Establishing s3 client")
        client = boto3.client(
            "s3",
            region_name=region_name,
            verify=verify_ssl,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            config=config,
        )
        log.info(f"Downloading S3 File: {bucket}/{key} to {local_path}")
        try:
            total = client.head_object(Bucket=bucket, Key=key)["ContentLength"]
            if show_progress:
                with ProgressPercentage(client, bucket, key, total) as Callback:
                    client.download_file(bucket, key, local_path, Callback=Callback)
            else:
                client.download_file(bucket, key, local_path)
        except ClientError:
            raise KeyError(f"File: {bucket}/{key} not available!!")
        return local_path


def upload(
    bucket: str,
    key: str,
    local_path: str,
    config: botocore.client.Config = None,
    region_name: str = None,
    aws_access_key_id: str = os.getenv("ARMORY_PRIVATE_S3_ID", default=None),
    aws_secret_access_key: str = os.getenv("ARMORY_PRIVATE_S3_KEY", default=None),
    verify_ssl: bool = False,
    make_public: bool = False,
    show_progress: bool = True,
):
    """Upload File from armory S3
    Parameters:
        bucket (str):                       The name of the s3 bucket to use
        key (str):                          The name of file in the bucket to download
        local_path (str):                   Where to put the downloaded file
        config (botocore.client.Config):    Boto Client Config.  For example if you want to download anonymously
                                             you can use `config=armory.utils.s3.ANONYMOUS_CONFIG`
                                             (default: None)
        region_name (str):                  AWS region for the client to use.
        aws_access_key_id (str):            AWS key ID for S3 access.  If `config=None` this should be set!
                                                (default: None)
        aws_secret_access_key (str):        AWS Secret key for S3 access.  If `config=None` this should be set!
                                                (default: None)
        verify_ssl (bool):                  Whether or not the client should verify SSL certificates
        make_public (bool):                 Set permission on s3 artifact to `public-read` (default: False)
        show_progress (bool):               Whether or not to use armory.logs.ProgressPercentage to show progress bar
                                            during download.

    Returns:
            None
    """
    log.debug("Establishing s3 client...")
    client = boto3.client(
        "s3",
        region_name=region_name,
        verify=verify_ssl,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        config=config,
    )
    log.info(f"Uploading File: {local_path} to S3: {bucket}/{key}")
    extra_args = {"ACL": "public-read"} if make_public else None
    if show_progress:
        client.upload_file(
            local_path,
            bucket,
            key,
            Callback=ProgressPercentageUpload(local_path),
            ExtraArgs=extra_args,
        )
    else:
        client.upload_file(local_path, bucket, key, ExtraArgs=extra_args)
