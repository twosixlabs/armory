from armory.datasets import common

import argparse


def load(name: str, version: str = "") -> (dict, dict):
    """
    Load the given dataset, optionally with specified version
        If version is not provided, the latest local version will be used

    By default, this loads from a base directory of ~/armory/dataset/new_builds

    Return (Info, Dataset) tuple
    """
    common.get_root()

    raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "name",
        type=str,
        help="dataset name",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="",
        help="specify the dataset version",
    )
    args = parser.parse_args()
    load(args.name, version=args.version)
