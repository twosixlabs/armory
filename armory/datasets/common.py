from pathlib import Path

from armory import paths

SUBDIR = "new_builds"


def get_root():
    return Path(paths.runtime_paths().dataset_dir) / SUBDIR
