"""
Automatic speech recognition model

Model contributed by: MITRE Corporation
"""

# BEGIN hacks
# Save deep speech model to armory
# This can be made less hacky after this ART issue:
# https://github.com/Trusted-AI/adversarial-robustness-toolbox/issues/693
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

from armory import paths

ART_DATA_PATH = os.path.join(paths.runtime_paths().saved_model_dir, "art")
os.makedirs(ART_DATA_PATH, exist_ok=True)
from art.estimators.speech_recognition import pytorch_deep_speech

pytorch_deep_speech.ART_DATA_PATH = ART_DATA_PATH
logger.warning(f"Saving art deep speech model weights to {ART_DATA_PATH}")
# END hacks

from art.estimators.speech_recognition import PyTorchDeepSpeech

# Workaround for ART 1.6.0, due to compute_loss issue
# TODO: revert to commented code in ART 1.6.1
# def get_art_model(
#     model_kwargs: dict, wrapper_kwargs: dict, weights_path: Optional[str] = None
# ) -> PyTorchDeepSpeech:
#     return PyTorchDeepSpeech(**wrapper_kwargs)


class PyTorchDeepSpeechModel(PyTorchDeepSpeech):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute_loss(self):
        raise NotImplementedError


def get_art_model(
    model_kwargs: dict, wrapper_kwargs: dict, weights_path: Optional[str] = None
) -> PyTorchDeepSpeech:
    return PyTorchDeepSpeechModel(**wrapper_kwargs)
