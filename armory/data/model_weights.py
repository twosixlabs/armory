"""
Utility to download model weights to cache.
"""
import os

from armory import paths
from armory.data.utils import _read_validate_scenario_config, download_file_from_s3
from armory.logs import log


def download_all(download_config, scenario):
    config = _read_validate_scenario_config(download_config)
    if scenario == "all":
        for scenario in config["scenario"].keys():
            for weights_file in config["scenario"][scenario]["weights_file"]:
                _download_weights(weights_file)
    elif scenario == "list":
        return
    else:
        for weights_file in config["scenario"][scenario]["weights_file"]:
            _download_weights(weights_file)


def _download_weights(weights_file, force_download=False):
    if not weights_file:
        return

    saved_model_dir = paths.runtime_paths().saved_model_dir
    filepath = os.path.join(saved_model_dir, weights_file)

    if os.path.isfile(filepath) and not force_download:
        log.info(f"Model weights file {filepath} found, skipping.")
    else:
        if os.path.isfile(filepath):
            log.info("Forcing overwrite of old file.")
            os.remove(filepath)

        log.info(f"Downloading weights file {weights_file} from s3...")

        download_file_from_s3(
            "armory-public-data",
            f"model-weights/{weights_file}",
            f"{saved_model_dir}/{weights_file}",
        )
