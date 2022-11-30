"""
Updated model for LibriSpeech ASR that doesn't require extra dependencies
"""

# https://pytorch.org/audio/stable/pipelines.html#torchaudio.pipelines.Wav2Vec2Bundle
from typing import List

from art.estimators.pytorch import PyTorchEstimator
import numpy as np
import torch
import torchaudio

# from torchaudio.models.decoder import ctc_decoder


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> List[str]:
        """Given a sequence emission over labels, get the best path
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.
        Returns:
          List[str]: The resulting transcript
        """
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = indices[indices != 0]
        joined = "".join([self.labels[i] for i in indices])
        return [joined.replace("|", " ").strip()]


def y_to_one_hot(y, labels):
    y_one_hot = []
    y_lengths = []
    for y_i in y:
        if isinstance(y_i, bytes):
            y_i = y_i.decode("utf-8")
        y_lengths.append(len(y_i))
        y_i = y_i.replace(" ", "|")
        y_one_hot.append([labels.index(char) for char in y_i])
    return torch.tensor(y_one_hot), tuple(y_lengths)


class HubertASRLarge(torch.nn.Module):
    def __init__(
        self,
        source_sample_rate=16000,
        blank=0,
        reduction="mean",
        zero_infinity=False,
        decoder="greedy",
    ):
        super().__init__()
        self.bundle = torchaudio.pipelines.HUBERT_ASR_LARGE
        self.model = self.bundle.get_model()
        self.labels = self.bundle.get_labels()
        self.source_sample_rate = source_sample_rate
        self.loss_fn = torch.nn.CTCLoss(
            blank=blank, reduction=reduction, zero_infinity=zero_infinity
        )
        if decoder == "greedy":
            self.decoder = GreedyCTCDecoder(self.labels)
        else:
            raise NotImplementedError("need to implement beam search decoder")
        self.model.eval()
        self.model.requires_grad_(False)

    def forward(self, x, transcript=True):
        if len(x) != 1:
            raise NotImplementedError("batch size should be 1")
        if self.bundle.sample_rate != self.source_sample_rate:
            x = torchaudio.functional.resample(
                x, self.source_sample_rate, self.bundle.sample_rate
            )

        emissions, _ = self.model(x)
        if transcript:
            return self.decoder(emissions)
        else:
            return emissions

    def x_grad(self, x: torch.Tensor, y: List[str]):
        if len(x) != 1:
            raise NotImplementedError("batch size should be 1")
        x.requires_grad = True
        target, target_lengths = y_to_one_hot(y, self.labels)
        input = self.forward(x, transcript=False).log_softmax(-1)
        input = input.transpose(0, 1)
        input_lengths = tuple([input.shape[0]])
        loss = self.loss_fn(input, target, input_lengths, target_lengths)
        loss.backward()
        return x.grad


class HubertASRLargeART(PyTorchEstimator):
    def __init__(self, device_type=DEVICE, **kwargs):
        for k in (
            "clip_values",
            "preprocessing_defences",
            "postprocessing_defences",
            "preprocessing",
        ):
            if kwargs.get(k) is not None:
                raise NotImplementedError(f"kwarg {k} not currently supported")
        super().__init__(
            model=None,
            clip_values=None,
            channels_first=None,
            preprocessing_defences=None,
            postprocessing_defences=None,
            preprocessing=None,
        )
        self._device = device_type
        self._model = HubertASRLarge()
        self._model.to(self._device)
        self._input_shape = None

    def predict(self, x: np.ndarray, transcription_output=True, batch_size=1):
        if len(x) != 1 or batch_size != 1:
            raise NotImplementedError("batch size should be 1")
        x = torch.tensor([i for i in x]).to(self._device)
        return self._model.forward(x, transcript=transcription_output)

    def loss_gradient(self, x: np.ndarray, y: np.ndarray):
        if len(x) != 1:
            raise NotImplementedError("batch size should be 1")
        x = torch.tensor([i for i in x]).to(self._device)
        grad = self._model.x_grad(x, y)
        return grad.detach().cpu().numpy()

    def get_activations(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def input_shape(self):
        return self._input_shape


def get_art_model(model_kwargs, wrapper_kwargs, weights_path):
    return HubertASRLargeART(**wrapper_kwargs)
