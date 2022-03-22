"""
Automatic speech recognition model

Model contributed by: MITRE Corporation
"""

from typing import Optional

from art.estimators.speech_recognition import PyTorchDeepSpeech


def get_art_model(
    model_kwargs: dict, wrapper_kwargs: dict, weights_path: Optional[str] = None
) -> PyTorchDeepSpeech:
    return PyTorchDeepSpeech(**wrapper_kwargs)
