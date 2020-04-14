"""
CNN model for raw audio classification

Model contributed by: MITRE Corporation
Adapted from: https://github.com/mravanelli/SincNet
"""
import logging

from art.classifiers import PyTorchClassifier
import numpy as np
import torch

from armory.data.utils import maybe_download_weights_from_s3

# Load model from MITRE external repo: https://github.com/hkakitani/SincNet
# This needs to be defined in your config's `external_github_repo` field to be
# downloaded and placed on the PYTHONPATH
from SincNet import dnn_models

logger = logging.getLogger(__name__)

SAMPLE_RATE = 8000
WINDOW_STEP_SIZE = 375
WINDOW_LENGTH = int(SAMPLE_RATE * WINDOW_STEP_SIZE / 1000)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocessing_fn(batch):
    """
    Standardize, then normalize sound clips
    """
    processed_batch = []
    for clip in batch:

        signal = clip.astype(np.float64)
        # Signal normalization
        signal = signal / np.max(np.abs(signal))

        # get random chunk of fixed length (from SincNet's create_batches_rnd)
        signal_length = len(signal)
        signal_start = np.random.randint(signal_length - WINDOW_LENGTH - 1)
        signal_stop = signal_start + WINDOW_LENGTH
        signal = signal[signal_start:signal_stop]
        processed_batch.append(signal)

    return np.array(processed_batch)


def sincnet(weights_file=None):
    pretrained = weights_file is not None
    if pretrained:
        filepath = maybe_download_weights_from_s3(weights_file)
        model_params = torch.load(filepath, map_location=DEVICE)
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


# NOTE: PyTorchClassifier expects numpy input, not torch.Tensor input
def get_art_model(model_kwargs, wrapper_kwargs, weights_file=None):
    model = sincnet(weights_file=weights_file, **model_kwargs)
    model.to(DEVICE)

    wrapped_model = PyTorchClassifier(
        model,
        loss=torch.nn.NLLLoss(),
        optimizer=torch.optim.RMSprop(
            model.parameters(), lr=0.001, alpha=0.95, eps=1e-8
        ),
        input_shape=(WINDOW_LENGTH,),
        nb_classes=40,
    )
    return wrapped_model
