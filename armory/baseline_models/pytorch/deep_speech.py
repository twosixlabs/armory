"""
Automatic speech recognition model

Model contributed by: MITRE Corporation
"""

import os
from typing import Optional

from art import config
from art.estimators.speech_recognition import PyTorchDeepSpeech

from armory import paths
from armory.utils.external_repo import ExternalRepoImport

# Test for external repo at import time to fail fast
with ExternalRepoImport(
    repo="SeanNaren/deepspeech.pytorch@V3.0",
    experiment="librispeech_asr_snr_undefended.json",
):
    from deepspeech_pytorch.model import DeepSpeech  # noqa: F401

config.set_data_path(os.path.join(paths.runtime_paths().saved_model_dir, "art"))


def get_art_model(
    model_kwargs: dict, wrapper_kwargs: dict, weights_path: Optional[str] = None
) -> PyTorchDeepSpeech:
    return PyTorchDeepSpeech(**wrapper_kwargs)
