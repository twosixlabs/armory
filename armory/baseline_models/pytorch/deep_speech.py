"""
Automatic speech recognition model

Model contributed by: MITRE Corporation
"""
import logging

from art.estimators.speech_recognition import PyTorchDeepSpeech
import numpy as np
import torch

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocessing_fn(batch):
    """
    Standardize, then normalize sound clips
    """
    processed_batch = []
    for clip in batch:
        signal = clip.astype(np.float32)
        # Signal normalization
        signal = signal / np.linalg.norm(signal, ord=2)
        processed_batch.append(signal)
    return np.array(processed_batch)


def get_art_model(model_kwargs, wrapper_kwargs, weights_file=None):
    wrapped_model = PyTorchDeepSpeech(**wrapper_kwargs,)
    return wrapped_model
