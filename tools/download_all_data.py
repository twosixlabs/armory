"""
Script to download all datasets and docker container for offline usage.

Running from project root:
    python -m tools.download_all_data
"""
import logging

import coloredlogs

from armory.docker.management import ManagementInstance


logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.INFO)


def dl_files():
    """
    Download all datasets to cache.
    """
    from armory.data.data import SUPPORTED_DATASETS

    for name, func in SUPPORTED_DATASETS.items():
        logger.info(f"Downloading (if necessary) dataset {name}")
        try:
            func()
        except Exception:
            logger.exception(f"Loading dataset {name} failed.")


def main():
    manager = ManagementInstance()
    runner = manager.start_armory_instance()
    runner.exec_cmd(
        "python -c 'from tools.download_all_data import dl_files; dl_files()'"
    )
    manager.stop_armory_instance(runner)


if __name__ == "__main__":
    main()
