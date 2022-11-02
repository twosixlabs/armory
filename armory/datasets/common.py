from functools import lru_cache
from pathlib import Path

import tensorflow_datasets as tfds

from armory import paths

SUBDIR = "new_builds"


def get_root():
    return Path(paths.runtime_paths().dataset_dir) / SUBDIR


@lru_cache
def tfds_builders() -> list:
    return tfds.list_builders()


@lru_cache
def armory_builders() -> list:
    source_root = Path(__file__).parent
    builders = {}
    for builder_dir in (source_root / "standard").iterdir():
        if builder_dir.is_dir():
            builders[builder_dir.stem] = str(builder_dir)
    for builder_dir in (source_root / "adversarial").iterdir():
        if builder_dir.is_dir():
            if builder_dir.stem in builders:
                log.warning(
                    f"{builder_dir.stem} is in both 'standard' and 'adversarial'. Ignoring adversarial duplicate."
                )
            else:
                builders[builder_dir.stem] = str(builder_dir)

    return builders
