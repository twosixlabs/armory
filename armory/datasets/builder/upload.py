"""
ARMORY Dataset Upload Script

This script is used to upload built datasets to the armory S3 datastore.

It also contains related utilities

To build the necessary datasets, see `build.py`.  Once built, this script
takes paths to the dataset directories, tarballs the directory, and
uploads that to s3.
"""

import argparse
import os
import sys
import tarfile
import tempfile
import threading

import boto3
import botocore.client
from loguru import logger as log

from armory.datasets.builder import utils


class ProgressPercentageUpload(object):
    """Helper Class for Progress Updates to stdout for uploads"""

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
        fpath = os.path.join(tmpdir, filename)
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


def main(directory, parent, bucket):
    # Getting List of all Dataset Directories
    data_dirs = utils.resolve_dataset_directories(directory, parent)

    log.info(f"Dataset Directories Selected: {data_dirs}")
    for ds in data_dirs:
        ds = ds.rstrip("/")
        log.info(f"Processing Dataset: {ds}")
        ds_info, ds = utils.load_from_directory(ds)
        fname = utils.get_dataset_archive_name(ds_info.name, ds_info.version)
        prepare_and_upload(fname, ds, bucket=bucket)


if __name__ == "__main__":
    epilog = "\n".join(
        [
            "To upload datasets [dir1] [dir2] ...  use:",
            "\t python upload.py [dir1] [dir2] ...",
            "Additionally, use `-p` `--parent` to upload all datasets in [data_dir] directory:",
            "\t python upload.py --parent [data_dir]",
            "\nNOTE: Each Dataset directory must be a `TFDS` style directory containing ",
            "`tfrecord` files and associated metadata files.",
        ]
    )
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        usage="%(prog)s [options]",
        epilog=epilog,
    )
    parser.add_argument("directory", nargs="*", help="TFDS Style Dataset Directory")
    parser.add_argument(
        "-p",
        "--parent",
        default=[],
        nargs="+",
        action="append",
        help="Parent Directory containing one or more TFDS Style Dataset directories",
    )
    parser.add_argument(
        "-v",
        "--verbosity",
        type=str,
        choices=["trace", "debug", "info", "warning", "error"],
        default="info",
        help="Set Output log level (Default: %(default)s)",
    )
    parser.add_argument(
        "-b",
        "--bucket",
        type=str,
        default="ds-noodle",
        help="Armory s3 Bucket to upload to",
    )
    args = parser.parse_args()

    # Setting up Logger to stdout with chosen level
    utils.setup_logger(
        level=args.verbosity.upper(),
        suppress_tfds_progress=True
        if args.verbosity in ["warning", "error"]
        else False,
    )

    main(args.directory, args.parent, args.bucket)
