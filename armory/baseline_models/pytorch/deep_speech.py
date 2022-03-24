"""
Automatic speech recognition model

Model contributed by: MITRE Corporation
"""

import os
from typing import Optional

from armory import paths
from art import config

config.set_data_path(os.path.join(paths.runtime_paths().saved_model_dir, "art"))
from art.estimators.speech_recognition import PyTorchDeepSpeech


def get_art_model(
    model_kwargs: dict, wrapper_kwargs: dict, weights_path: Optional[str] = None
) -> PyTorchDeepSpeech:
    return PyTorchDeepSpeech(**wrapper_kwargs)
