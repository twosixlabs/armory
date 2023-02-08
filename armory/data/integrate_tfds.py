# Tensorflow dataset integration script
# For more details see docs/integrate_tensorflow_datasets.md
# in the armory repo or the "Integrating TFDS Datasets" section on
# armory.readthedocs.io

import os
import subprocess
import sys

import tensorflow_datasets as tfds

from armory import paths
from armory.data.datasets import CACHED_CHECKSUMS_DIR, _parse_dataset_name
from armory.data.template_boilerplate import fn_template
from armory.data.utils import sha256, upload_file_to_s3
from armory.logs import log


def main():
    k1 = "ARMORY_PRIVATE_S3_ID"
    k2 = "ARMORY_PRIVATE_S3_KEY"
    aws_access_key_id = os.getenv(k1)
    aws_secret_access_key = os.getenv(k2)

    assert len(aws_access_key_id) > 0, f"Need to set AWS ID in shell variable {k1}"
    assert len(aws_secret_access_key) > 0, f"Need to set AWS key in shell variable {k2}"

    if len(sys.argv) != 2:
        raise ValueError("Need argument with tfds name!")

    ds_name = sys.argv[1]

    dataset_dir = paths.runtime_paths().dataset_dir
    log.info("Preparing dataset (may take some time)...")
    ds = tfds.load(ds_name, data_dir=dataset_dir)
    assert len(ds) > 0

    name, subpath = _parse_dataset_name(ds_name)

    expected_path = os.path.join(dataset_dir, name, subpath)
    if not os.path.isdir(expected_path):
        raise ValueError(f"Dataset {ds_name} not found at {expected_path}!")

    tar_filepath = ds_name.replace(":", "_").replace("/", "_") + ".tar.gz"
    tar_full_filepath = os.path.join(dataset_dir, tar_filepath)

    log.info("Creating tarball (may take some time)...")
    completedprocess = subprocess.run(
        [
            "tar",
            "cvzf",
            tar_full_filepath,
            name,
        ],
        cwd=dataset_dir,
    )
    if completedprocess.returncode:
        raise Exception("bash tar failed. Please manually tar file and upload to S3")

    log.info("Uploading tarball...")
    upload_file_to_s3(f"{name}/{tar_filepath}", tar_full_filepath, public=True)

    size = os.path.getsize(tar_full_filepath)
    hash = sha256(tar_full_filepath)

    checksum_filename = os.path.join(CACHED_CHECKSUMS_DIR, f"{name}.txt")
    with open(checksum_filename, "w+") as fh:
        fh.write(f"armory-public-data {name}/{tar_filepath} {size} {hash}\n")

    template_filename = f"TEMPLATE_{name}.txt"
    with open(template_filename, "w+") as fh:
        fh.write(fn_template.replace("{name}", name).replace("{ds_name}", ds_name))
        log.info(
            "Template with boilerplate to update "
            f"armory/data/datasets.py located at {template_filename}..."
        )


if __name__ == "__main__":
    main()
