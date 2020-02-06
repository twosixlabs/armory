"""
Script to download all datasets and docker container for offline usage.

Running from project root:
    python -m tools.download_all_data
"""
from armory.docker.management import ManagementInstance


def dl_files():
    from armory.data.data import SUPPORTED_DATASETS

    for _, func in SUPPORTED_DATASETS.items():
        func()


if __name__ == "__main__":
    manager = ManagementInstance()
    runner = manager.start_armory_instance()
    runner.exec_cmd(
        "python -c 'from tools.download_all_data import dl_files; dl_files()'"
    )
    manager.stop_armory_instance(runner)
