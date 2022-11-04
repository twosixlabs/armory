from pathlib import Path
import os
import random
import shutil
import string
import subprocess

from armory.logs import log
from armory.datasets import build, common


def package(
    name,
    version: str = None,
    data_dir: str = None,
    cache_subdir=common.CACHE_SUBDIR,
    overwrite: bool = False,
) -> str:
    """
    Package a built dataset as .tar.gz, return path
    """
    version, data_dir, built_data_dir, subdir = build.build_info(
        name, version=version, data_dir=data_dir
    )

    data_dir = Path(data_dir)
    expected_dir = data_dir / name / version
    if not expected_dir.is_dir():
        raise FileNotFoundError(f"Dataset {name} not found at {expected_dir}")
    tar_full_filepath = common.get_cache_dataset_path(name, version)

    if tar_full_filepath.is_file():
        if overwrite:
            tar_full_filepath.unlink(missing_ok=True)
        else:
            raise FileExistsError(
                f"Dataset {name} cache file {tar_full_filepath} exists. Use overwrite=True"
            )

    log.info("Creating tarball (may take some time)...")
    cmd = ["tar", "cvzf", str(tar_full_filepath), str(Path(name) / version)]
    log.info(f"Running {' '.join(cmd)}")
    completed_process = subprocess.run(cmd, cwd=data_dir)
    completed_process.check_returncode()
    return str(tar_full_filepath)


def update(name, version: str = None, data_dir: str = None, url=None):
    """
    Hash file and update cached datasets file
    """
    version, data_dir, built_data_dir, subdir = build.build_info(
        name, version=version, data_dir=data_dir
    )
    filepath = common.get_cache_dataset_path(name, version)
    if not filepath.is_file():
        raise FileNotFoundError(f"filepath '{filepath}' not found.")
    assert (name, version) == common.parse_cache_filename(filepath.name)
    subdir = str(Path(name) / version)
    file_size, file_sha256 = common.hash_file(filepath)

    common.update_cached_datasets(name, version, subdir, file_size, file_sha256, url)


def verify(name, data_dir: str = None):
    info = common.cached_datasets()[name]
    version = info["version"]

    filepath = common.get_cache_dataset_path(name, version)
    if not filepath.is_file():
        raise FileNotFoundError(f"filepath '{filepath}' for dataset {name} not found.")

    common.verify_hash(filepath, info["size"], info["sha256"])


def extract(name, data_dir: str = None, overwrite: bool = False):
    """
    Extract cached dataset into tmp file then merge into data_dir
    """
    info = common.cached_datasets()[name]
    version = info["version"]

    if data_dir is None:
        data_dir = common.get_root()
    cache_dir = common.get_cache_dir(data_dir)
    filepath = common.get_cache_dataset_path(name, version)
    if not filepath.is_file():
        raise FileNotFoundError(f"filepath '{filepath}' for dataset {name} not found.")

    target_data_dir = data_dir / name / version
    if target_data_dir.exists() and not overwrite:
        raise ValueError("Target directory exists. Set overwrite=True to overwrite")

    # Extract to tmp directory
    tmp_dir = Path(cache_dir) / (
        "tmp_" + "".join(random.choice(string.ascii_lowercase) for _ in range(16))
    )
    tmp_dir.mkdir()
    cmd = ["tar", "zxvf", str(filepath), "--directory", str(tmp_dir)]
    log.info(f"Running {' '.join(cmd)}")
    completed_process = subprocess.run(cmd)
    completed_process.check_returncode()

    # should have directory structure <tmp_dir>/<name>/<version>/<data>
    if len(os.listdir(tmp_dir)) != 1:
        raise ValueError(f"{tmp_dir} has more than 1 directory inside")
    if name not in os.listdir(tmp_dir):
        raise ValueError(f"{name} does not match directory in {tmp_dir}")
    tmp_dir_name = tmp_dir / name

    if len(os.listdir(tmp_dir_name)) != 1:
        raise ValueError(f"{tmp_dir_name} has more than 1 directory inside")
    if version not in os.listdir(tmp_dir_name):
        raise ValueError(f"{version} does not match directory in {tmp_dir_name}")
    source_data_dir = tmp_dir_name / version

    if any(child.is_dir() for child in source_data_dir.iterdir()):
        raise ValueError("Data directory should not have subdirectories")

    if target_data_dir.exists() and overwrite:
        shutil.rmtree(target_data_dir)
    os.makedirs(target_data_dir.parent, exist_ok=True)
    shutil.move(source_data_dir, target_data_dir)
    shutil.rmtree(tmp_dir)
