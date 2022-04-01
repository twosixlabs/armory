import tempfile

from loguru import logger as log
import os
import tensorflow_datasets as tfds
import boto3
import botocore.client
import threading
import sys
import tarfile
import pathlib


def get_ds_path(dataset_name: str, dataset_directory: str) -> str:
    """Return the Dataset Path
    Parameters:
        dataset_name (str):         Name of the Dataset/Directory (e.g. `mnist`)
        dataset_directory (str):    Directory containing the Dataset

    Returns:
        dataset_path (str):         This will be the full path to the dataset contents
                                    which will include the dataset version.  For example
                                    `mnist` would return [data_dir]/mnist/3.0.1/
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
    return os.path.join(ds_path, versions[0])


def get_ds_archive_name(dataset_name: str, dataset_directory: str) -> str:
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
    fname = pathlib.Path(os.path.join(dataset_name, versions[0]))
    fname = "_".join(fname.parts)
    return fname


def load(dataset_name: str, dataset_directory: str):
    """Loads the TFDS Dataset using `tfds.core.builder_from_directory` method
    Parameters:
        dataset_name (str):         Name of the Dataset/Directory (e.g. `mnist`)
        dataset_directory (str):    Directory containing the Dataset

    Returns:
        ds_info (obj):              The TFDS dataset info JSON object.
        ds (tfds.core.Dataset)      The TFDS dataset
    """

    ds_path = get_ds_path(dataset_name, dataset_directory)
    expected_name = ds_path.replace(f"{dataset_directory}/", "")
    if not os.path.isdir(ds_path):
        raise ValueError(
            f"Dataset Directory: {ds_path} does not exist...cannot construct!!"
        )
    log.info(
        f"Attempting to Load Dataset: {dataset_name} from local directory: {dataset_directory}"
    )
    log.debug("Generating Builder object...")
    builder = tfds.core.builder_from_directory(ds_path)
    log.debug(
        f"Dataset Full Name: `{builder.info.full_name}`  Expected: `{expected_name}`"
    )
    if expected_name != builder.info.full_name:
        raise RuntimeError(
            f"Dataset Full Name: {builder.info.full_name}  differs from expected: {expected_name}"
            "...make sure that the build_class_file name matches the class name!!"
            "NOTE:  tfds converts camel case class names to lowercase separated by `_`"
        )
    log.debug("Converting to dataset")
    ds = builder.as_dataset()

    log.success("Loading Complete!!")
    return builder.info, ds


class ProgressPercentageUpload(object):
    def __init__(self, filename):
        self._filename = filename
        self._size = float(os.path.getsize(filename))
        self._seen_so_far = 0
        self._lock = threading.Lock()

    def __call__(self, bytes_amount):
        # To simplify, assume this is hooked up to a single filename
        with self._lock:
            self._seen_so_far += bytes_amount
            percentage = (self._seen_so_far / self._size) * 100
            sys.stdout.write(
                "\r%s  %s / %s  (%.2f%%)"
                % (self._filename, self._seen_so_far, self._size, percentage)
            )
            sys.stdout.flush()
            if int(percentage) == 100:
                sys.stdout.write("\n")


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


def prepare_and_upload(filename: str, local_path: str, bucket: str):
    with tempfile.TemporaryDirectory() as tmpdir:
        fpath = os.path.join(tmpdir, f"{filename}.tar.gz")
        log.info(f"Creating TAR Archive at: {fpath}")
        with tarfile.open(fpath, "w:gz") as tar:
            tar.add(local_path, arcname=os.path.basename(local_path))
        log.info("Uploading File to S3")
        upload(
            bucket=bucket,
            key=os.path.basename(fpath),
            local_path=fpath,
            verify_ssl=True,
        )
        log.success(f"Upload Complete!  See: s3://{bucket}/{os.path.basename(fpath)}")
