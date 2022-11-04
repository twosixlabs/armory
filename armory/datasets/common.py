from functools import lru_cache
import json
import hashlib
from pathlib import Path

import tensorflow_datasets as tfds

from armory import paths
from armory.logs import log

SUBDIR = "new_builds"
CACHE_JSON = "cached_datasets.json"
CACHE_SUBDIR = "cache"
_CACHED_DATASETS = None
DELIM = "__"
EXT = "cache.tar.gz"

ARMORY_PUBLIC_DATA = "armory-public-data"
ARMORY_PRIVATE_DATA = "armory-private-data"


def get_bucket(public: bool = False):
    if public:
        return ARMORY_PUBLIC_DATA
    else:
        return ARMORY_PRIVATE_DATA


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


def cached_datasets() -> dict:
    global _CACHED_DATASETS
    if _CACHED_DATASETS is None:
        json_file = Path(__file__).parent / CACHE_JSON
        with open(json_file) as f:
            _CACHED_DATASETS = json.load(f)
    return _CACHED_DATASETS


def update_cached_datasets(name, version, subdir, size, sha256, url):
    cache = cached_datasets()
    cache[name] = dict(
        sha256=sha256,
        size=size,
        subdir=subdir,
        url=url,
        version=version,
    )
    json_file = Path(__file__).parent / CACHE_JSON
    with open(json_file, "w") as f:
        f.write(json.dumps(cache, sort_keys=True, indent=4) + "\n")
    global _CACHED_DATASETS
    _CACHED_DATASETS = None  # force cached_datasets() to reload when called
    log.info("Cached datasets updated. Ensure the update is added to git")


def size(filepath: str):
    return Path(filepath).stat().st_size


def verify_size(filepath: str, file_size: int):
    actual_size = size(filepath)
    if actual_size != file_size:
        raise ValueError(f"file size of {filepath}: {actual_size} != {file_size}")


def sha256(filepath: str, block_size=4096):
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(block_size), b""):
            sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()


def verify_sha256(filepath: str, hash_value: str, block_size: int = 4096):
    """
    Verify that the target filepath has the given sha256 hash_value
        Raise ValueError if False

    filepath - target filepath
    hash_value - hex encoded value of the hash
    block_size - block size for chunked reading from file
    """

    if len(hash_value) != 64:
        raise ValueError(f"Invalid hash_value: len({hash_value}) != 64")
    hash_value = hash_value.lower()
    if not all(x in "0123456789abcdef" for x in hash_value):
        raise ValueError(f"Invalid hash_value: {hash_value} contains non-hex chars")

    value = sha256(filepath, block_size=block_size)
    if value != hash_value:
        raise ValueError(f"sha256 hash of {filepath}: {value} != {hash_value}")


def hash_file(path: str) -> (int, str):
    """
    Convenience function. Return size and sha256 of file
    """
    return size(path), sha256(path)


def verify_hash(filepath: str, file_size: int, hash_value: str):
    """
    Convenience function. Verify size and sha256 of file
    """
    verify_size(filepath, file_size)
    verify_sha256(filepath, hash_value)


def generate_cache_filename(name, version, delim=DELIM, ext=EXT):
    return delim.join([str(name), str(version), ext])


def parse_cache_filename(filename, delim=DELIM, ext=EXT):
    tokens = filename.split(delim)
    if len(tokens) == 3 and tokens[2] == ext:
        name, version = tokens[:2]
    else:
        form = generate_cache_filename("<name>", "<version")
        raise ValueError(f"{filename} not in {form} cache format")
    return name, version


def get_cache_dir(data_dir):
    cache_dir = Path(data_dir) / CACHE_SUBDIR
    cache_dir.mkdir(exist_ok=True)
    return cache_dir


def get_cache_dataset_path(name, version, data_dir: str = None):
    if data_dir is None:
        data_dir = get_root()
    filepath = get_cache_dir(data_dir) / generate_cache_filename(name, version)
    return filepath


def get_cache_key(name, version):
    """
    Get the s3 key for a corresponding cache file
    """
    return str(Path("datasets") / "cache" / generate_cache_filename(name, version))


def get_cache_url(name, version, public=False):
    """
    Get the s3 url for a corresponding cache file
    """
    bucket = get_bucket(public=public)
    key = get_cache_key(name, version)
    return f"s3://{bucket}/{key}"
