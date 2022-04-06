"""ARMORY Dataset Upload Script

This script is used to upload built datasets to the armory S3 datastore.
To build the necessary datasets, see `build.py`.  Once built, this script
takes paths to the dataset directories, tarballs the directory, and
uploads that to s3.

"""
from loguru import logger as log
import utils


if __name__ == "__main__":
    import argparse

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

    # Getting List of all Dataset Directories
    data_dirs = utils.resolve_dataset_directories(args.directory, args.parent)

    log.info(f"Dataset Directories Selected: {data_dirs}")
    for ds in data_dirs:
        ds = ds.rstrip("/")
        log.info(f"Processing Dataset: {ds}")
        ds_info, ds = utils.load_from_directory(ds)
        fname = utils.get_dataset_archive_name(ds_info.name, ds_info.version)
        utils.prepare_and_upload(fname, ds, bucket=args.bucket)
