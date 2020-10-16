"""
Automatic speech recognition model

Model contributed by: MITRE Corporation
"""

from art.estimators.speech_recognition import PyTorchDeepSpeech
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_art_model(model_kwargs, wrapper_kwargs, weights_file=None):
    return PyTorchDeepSpeech(**wrapper_kwargs)
