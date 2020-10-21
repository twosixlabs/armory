"""
Automatic speech recognition model

Model contributed by: MITRE Corporation
"""

from art.estimators.speech_recognition import PyTorchDeepSpeech


def get_art_model(model_kwargs, wrapper_kwargs, weights_path=None):
    return PyTorchDeepSpeech(**wrapper_kwargs)
