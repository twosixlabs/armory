from pathlib import Path

from armory.data.utils import maybe_download_weights_from_s3
from armory.logs import log

PARENT_DIR = Path(__file__).parent


def get_path(filename) -> str:
    """
    Get the absolute path of the provided file name. Ordering priority is:
        1) Check directly for provided filepath
        2) Load from parent directory
        3) Attempt to download from s3 as a weights file
    """
    filename = Path(filename)
    if filename.is_file():
        return str(filename)
    filepath = PARENT_DIR / filename
    if filepath.is_file():
        return str(filepath)

    return maybe_download_weights_from_s3(filename)
