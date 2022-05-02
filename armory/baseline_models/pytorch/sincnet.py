"""
CNN model for raw audio classification

Model contributed by: MITRE Corporation
Adapted from: https://github.com/mravanelli/SincNet
"""
from typing import Optional

from art.estimators.classification import PyTorchClassifier
import numpy as np
import torch
from torch import nn

from armory.utils.external_repo import ExternalRepoImport

with ExternalRepoImport(
    repo="hkakitani/SincNet",
    experiment="librispeech_baseline_sincnet.json",
):
    from SincNet import dnn_models

# NOTE: Underlying dataset sample rate is 16 kHz. SincNet uses this SAMPLE_RATE to
# determine internal filter high cutoff frequency.
SAMPLE_RATE = 8000
WINDOW_STEP_SIZE = 375
WINDOW_LENGTH = int(SAMPLE_RATE * WINDOW_STEP_SIZE / 1000)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def numpy_random_preprocessing_fn(batch: np.ndarray):
    """
    Standardize, then normalize sound clips

    Then generate a random cut of the input
    """
    processed_batch = []
    for clip in batch:
        # convert and normalize
        signal = clip.astype(np.float32)
        # Signal normalization
        signal = signal / np.max(np.abs(signal))

        # make a pseudorandom cut of size equal to WINDOW_LENGTH
        # (from SincNet's create_batches_rnd)
        signal_length = len(signal)
        np.random.seed(signal_length)
        signal_start = int(
            np.random.randint(signal_length / WINDOW_LENGTH - 1)
            * WINDOW_LENGTH
            % signal_length
        )
        signal_stop = signal_start + WINDOW_LENGTH
        signal = signal[signal_start:signal_stop]
        processed_batch.append(signal)

    return np.array(processed_batch)


def numpy_all_preprocessing_fn(batch: np.ndarray):
    """
    Input is comprised of one or more clips, where each clip i
    is given as an ndarray with shape (n_i,).
    Preprocessing normalizes each clip and breaks each clip into an integer number
    of non-overlapping segments of length WINDOW_LENGTH.
    Output is a list of clips, each of shape (int(n_i/WINDOW_LENGTH), WINDOW_LENGTH)
    """
    if len(batch) != 1:
        raise NotImplementedError(
            "Requires ART variable length input capability for batch size != 1"
        )
    processed_batch = []
    for clip in batch:
        # convert and normalize
        signal = clip.astype(np.float64)
        signal = signal / np.max(np.abs(signal))

        # break into a number of chunks of equal length
        num_chunks = int(len(signal) / WINDOW_LENGTH)
        signal = signal[: num_chunks * WINDOW_LENGTH]
        signal = np.reshape(signal, (num_chunks, WINDOW_LENGTH), order="C")
        processed_batch.append(signal)
    # remove outer batch (of size 1)
    processed_batch = processed_batch[0]
    return np.array(processed_batch)


def torch_random_preprocessing_fn(x):
    """
    Standardize, then normalize sound clips
    """
    if x.shape[0] != 1:
        raise ValueError(f"Shape of batch x {x.shape[0]} != 1")
    if x.dtype != torch.float32:
        raise ValueError(f"dtype of batch x {x.dtype} != torch.float32")
    if x.max() > 1.0:
        raise ValueError(f"batch x max {x.max()} > 1.0")
    if x.min() < -1.0:
        raise ValueError(f"batch x min {x.min()} < -1.0")
    x = x.squeeze(0)

    # Signal normalization
    x = x / x.abs().max()

    # get pseudorandom chunk of fixed length (from SincNet's create_batches_rnd)
    signal_length = len(x)
    np.random.seed(signal_length)
    start = int(
        np.random.randint(signal_length / WINDOW_LENGTH - 1)
        * WINDOW_LENGTH
        % signal_length
    )

    x = x[start : start + WINDOW_LENGTH]

    x = x.unsqueeze(0)
    return x


def torch_all_preprocessing_fn(x: torch.Tensor):
    """
    Input is comprised of one or more clips, where each clip i
    is given as an ndarray with shape (n_i,).
    Preprocessing normalizes each clip and breaks each clip into an integer number
    of non-overlapping segments of length WINDOW_LENGTH.
    Output is a list of clips, each of shape (int(n_i/WINDOW_LENGTH), WINDOW_LENGTH)
    """
    if x.shape[0] != 1:
        raise NotImplementedError(
            "Requires ART variable length input capability for batch size != 1"
        )
    if x.max() > 1.0:
        raise ValueError(f"batch x max {x.max()} > 1.0")
    if x.min() < -1.0:
        raise ValueError(f"batch x min {x.min()} < -1.0")
    if x.dtype != torch.float32:
        raise ValueError(f"dtype of batch x {x.dtype} != torch.float32")
    x = x.squeeze(0)

    # Signal normalization
    x = x / x.abs().max()

    # break into a number of chunks of equal length
    num_chunks = int(len(x) / WINDOW_LENGTH)
    x = x[: num_chunks * WINDOW_LENGTH]
    x = x.reshape((num_chunks, WINDOW_LENGTH))

    return x


def sincnet(weights_path: Optional[str] = None) -> dnn_models.SincWrapper:
    """
    Set configuration options and instantiates SincWrapper object
    """
    pretrained = weights_path is not None
    if pretrained:
        model_params = torch.load(weights_path, map_location=DEVICE)
    else:
        model_params = {}
    CNN_params = model_params.get("CNN_model_par")
    DNN1_params = model_params.get("DNN1_model_par")
    DNN2_params = model_params.get("DNN2_model_par")

    # from SincNet/cfg/SincNet_dev_LibriSpeech.cfg
    cnn_N_filt = [80, 60, 60]
    cnn_len_filt = [251, 5, 5]
    cnn_max_pool_len = [3, 3, 3]
    cnn_use_laynorm_inp = True
    cnn_use_batchnorm_inp = False
    cnn_use_laynorm = [True, True, True]
    cnn_use_batchnorm = [False, False, False]
    cnn_act = ["relu", "relu", "relu"]
    cnn_drop = [0.0, 0.0, 0.0]

    fc_lay = [2048, 2048, 2048]
    fc_drop = [0.0, 0.0, 0.0]
    fc_use_laynorm_inp = True
    fc_use_batchnorm_inp = False
    fc_use_batchnorm = [True, True, True]
    fc_use_laynorm = [False, False, False]
    fc_act = ["leaky_relu", "linear", "leaky_relu"]

    class_lay = [40]
    class_drop = [0.0, 0.0]
    class_use_laynorm_inp = True
    class_use_batchnorm_inp = False
    class_use_batchnorm = [False]
    class_use_laynorm = [False]
    class_act = ["softmax"]

    CNN_options = {
        "input_dim": WINDOW_LENGTH,
        "fs": SAMPLE_RATE,
        "cnn_N_filt": cnn_N_filt,
        "cnn_len_filt": cnn_len_filt,
        "cnn_max_pool_len": cnn_max_pool_len,
        "cnn_use_laynorm_inp": cnn_use_laynorm_inp,
        "cnn_use_batchnorm_inp": cnn_use_batchnorm_inp,
        "cnn_use_laynorm": cnn_use_laynorm,
        "cnn_use_batchnorm": cnn_use_batchnorm,
        "cnn_act": cnn_act,
        "cnn_drop": cnn_drop,
        "pretrained": pretrained,
        "model_params": CNN_params,
    }

    DNN1_options = {
        "fc_lay": fc_lay,
        "fc_drop": fc_drop,
        "fc_use_batchnorm": fc_use_batchnorm,
        "fc_use_laynorm": fc_use_laynorm,
        "fc_use_laynorm_inp": fc_use_laynorm_inp,
        "fc_use_batchnorm_inp": fc_use_batchnorm_inp,
        "fc_act": fc_act,
        "pretrained": pretrained,
        "model_params": DNN1_params,
    }

    DNN2_options = {
        "input_dim": fc_lay[-1],
        "fc_lay": class_lay,
        "fc_drop": class_drop,
        "fc_use_batchnorm": class_use_batchnorm,
        "fc_use_laynorm": class_use_laynorm,
        "fc_use_laynorm_inp": class_use_laynorm_inp,
        "fc_use_batchnorm_inp": class_use_batchnorm_inp,
        "fc_act": class_act,
    }

    sincNet = dnn_models.SincWrapper(DNN2_options, DNN1_options, CNN_options)

    if pretrained:
        sincNet.eval()
        sincNet.load_state_dict(DNN2_params)

    else:
        sincNet.train()

    return sincNet


class SincNetWrapper(nn.Module):
    MODES = {
        "random": torch_random_preprocessing_fn,
        "all": torch_all_preprocessing_fn,
    }

    def __init__(self, model_kwargs: dict, weights_path: Optional[str]) -> None:
        super().__init__()
        predict_mode = model_kwargs.pop("predict_mode", "all")
        if predict_mode not in self.MODES:
            raise ValueError(f"predict_mode {predict_mode} not in {tuple(self.MODES)}")
        self.predict_mode = predict_mode

        self.model = sincnet(weights_path=weights_path, **model_kwargs)
        self.model.to(DEVICE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # preprocessing should be done before model for arbitrary length input
            return self.model(x)

        x = self.MODES[self.predict_mode](x)
        output = self.model(x)
        if self.predict_mode == "all":
            output = torch.mean(output, dim=0, keepdim=True)
        return output


preprocessing_fn = numpy_random_preprocessing_fn


def get_art_model(
    model_kwargs: dict, wrapper_kwargs: dict, weights_path: Optional[str] = None
) -> PyTorchClassifier:
    model = SincNetWrapper(model_kwargs, weights_path)
    model.to(DEVICE)

    wrapped_model = PyTorchClassifier(
        model,
        loss=torch.nn.NLLLoss(),
        optimizer=torch.optim.RMSprop(
            model.parameters(), lr=0.001, alpha=0.95, eps=1e-8
        ),
        input_shape=(None,),
        nb_classes=40,
        **wrapper_kwargs,
    )
    return wrapped_model
