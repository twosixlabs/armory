from pathlib import Path

from armory.data.utils import maybe_download_weights_from_s3
from armory.logs import log

TRIGGERS_DIR = Path(__file__).parent


def get_path(filename) -> str:
    """
    Get the absolute path of the provided trigger. Ordering priority is:
        1) Check directly for provided filepath
        2) Load from `triggers` directory
        3) Attempt to download from s3 as a weights file
    """
    filename = Path(filename)
    if filename.is_file():
        return str(filename)
    triggers_path = TRIGGERS_DIR / filename
    if triggers_path.is_file():
        return str(triggers_path)

    return maybe_download_weights_from_s3(filename)
