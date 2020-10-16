"""
Automatic speech recognition model

Model contributed by: MITRE Corporation
"""
import logging
from typing import Optional, Tuple

from art.estimators.speech_recognition import PyTorchDeepSpeech
from art.defences.preprocessor.preprocessor import PreprocessorPyTorch
import numpy as np  # TODO: remove
import torch

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#x_preprocessed# TODO: move preprocessing_fn to inside model!
#def preprocessing_fn(batch):
#    """
#    Standardize, then normalize sound clips
#    """
#    processed_batch = []
#    for clip in batch:
#        signal = clip.astype(np.float32)
#        # Signal normalization
#        signal = signal / np.linalg.norm(signal, ord=2)
#        processed_batch.append(signal)
#    return np.array(processed_batch)


class NormalizeSignal(PreprocessorPyTorch):
    def forward(self, x, y=None):
        x_preprocessed = torch.stack([x_i / torch.norm(x_i, p=2) for x_i in x])
        return x_preprocessed, y

    def estimate_forward(self, x, y=None):
        x_preprocessed, y = self.forward(x, y=y)
        return x_preprocessed

    def apply_fit(self) -> bool:
        return True

    def apply_predict(self) -> bool:
        return True

    def __call__(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        raise NotImplementedError("Should only use PyTorch processing")

    def fit(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> None:
        raise NotImplementedError

    def estimate_gradient(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Should only use PyTorch processing")


def get_art_model(model_kwargs, wrapper_kwargs, weights_file=None):
    wrapped_model = PyTorchDeepSpeech(
        preprocessing_defences=NormalizeSignal(),
        **wrapper_kwargs
    )
    return wrapped_model
