from pathlib import Path
import os
import random
import shutil
import string
import subprocess

from armory.logs import log
from armory.datasets import build, common, upload


def package(
    name,
    version: str = None,
    data_dir: str = None,
    overwrite: bool = False,
) -> str:
    """
    Package a built dataset as .tar.gz, return path
    """
    version, data_dir, built_data_dir, subdir, builder_configs = build.build_info(
        name, version=version, data_dir=data_dir
    )

    # print(builder_configs)
    data_dir = Path(data_dir)

    if not builder_configs:
        tar_list = [str(Path(name) / version)]
    else:
        # metadata.json contains default_config_name for the given dataset
        tar_list = [
            str(Path(name) / config.name / version) for config in builder_configs
        ] + [str(Path(name) / ".config")]

    for tar_path in tar_list:
        expected_dir = data_dir / tar_path
        if not expected_dir.is_dir():
            # raise FileNotFoundError(f"Dataset {name} not found at {expected_dir}")
            raise FileNotFoundError(f"Dataset {tar_path} not found at {expected_dir}")

    tar_full_filepath = common.get_cache_dataset_path(name, version)
    if tar_full_filepath.is_file():
        if overwrite:
            tar_full_filepath.unlink(missing_ok=True)
        else:
            raise FileExistsError(
                f"Dataset {name} cache file {tar_full_filepath} exists. Use overwrite=True"
            )

    cmd = ["tar", "cvzf", str(tar_full_filepath)] + tar_list

    log.info("Creating tarball (may take some time)...")
    log.info(f"Running {' '.join(cmd)}")
    completed_process = subprocess.run(cmd, cwd=data_dir)
    completed_process.check_returncode()
    return str(tar_full_filepath)


def update(name, version: str = None, data_dir: str = None, url=None):
    """
    Hash file and update cached datasets file
    """
    version, data_dir, built_data_dir, subdir, builder_configs = build.build_info(
        name, version=version, data_dir=data_dir
    )
    filepath = common.get_cache_dataset_path(name, version)
    if not filepath.is_file():
        raise FileNotFoundError(f"filepath '{filepath}' not found.")
    assert (name, version) == common.parse_cache_filename(filepath.name)

    if not builder_configs:
        subdir = str(Path(name) / version)
    else:
        subdir = [str(Path(name) / config.name / version) for config in builder_configs]
    file_size, file_sha256 = common.hash_file(filepath)

    common.update_cached_datasets(name, version, subdir, file_size, file_sha256, url)


def verify(name, data_dir: str = None):
    info = common.cached_datasets()[name]
    version = info["version"]

    filepath = common.get_cache_dataset_path(name, version, data_dir=data_dir)
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


def add_to_cache(
    name,
    version: str = None,
    data_dir: str = None,
    overwrite: bool = False,
    public: bool = False,
):
    """
    Convenience function for packaging, uploading, and adding to cache
    """
    package(
        name,
        version=version,
        data_dir=data_dir,
        overwrite=overwrite,
    )
    update(name, version=version, data_dir=data_dir)
    verify(name, data_dir=data_dir)
    upload.upload(name, public=public)
