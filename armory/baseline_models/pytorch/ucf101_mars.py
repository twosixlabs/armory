"""
CNN model for 241x100x1 audio spectrogram classification

Model contributed by: MITRE Corporation
Adapted from: https://github.com/craston/MARS
"""
import logging

from art.classifiers import PyTorchClassifier
import numpy as np
from PIL import Image
import torch
from torch import optim

# Load model from MITRE external repo: https://github.com/yusong-tan/MARS
# This needs to be defined in your config's `external_github_repo` field to be
# downloaded and placed on the PYTHONPATH
from MARS.opts import parse_opts
from MARS.models.model import generate_model
from MARS.dataset import preprocess_data

from armory.data.utils import maybe_download_weights_from_s3


logger = logging.getLogger(__name__)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_model(model_status="ucf101_trained", weights_file=None):
    statuses = ("ucf101_trained", "kinetics_pretrained")
    if model_status not in statuses:
        raise ValueError(f"model_status {model_status} not in {statuses}")
    trained = model_status == "ucf101_trained"
    if not trained and weights_file is None:
        raise ValueError("weights_file cannot be None for 'kinetics_pretrained'")

    if weights_file:
        filepath = maybe_download_weights_from_s3(weights_file)

    opt = parse_opts(arguments=[])
    opt.dataset = "UCF101"
    opt.only_RGB = True
    opt.log = 0
    opt.batch_size = 1
    opt.arch = f"{opt.model}-{opt.model_depth}"

    if trained:
        opt.n_classes = 101
    else:
        opt.n_classes = 400
        opt.n_finetune_classes = 101
        opt.batch_size = 32
        opt.ft_begin_index = 4

        opt.pretrain_path = filepath

    logger.info(f"Loading model... {opt.model} {opt.model_depth}")
    model, parameters = generate_model(opt)

    if trained and weights_file is not None:
        checkpoint = torch.load(filepath, map_location=DEVICE)
        model.load_state_dict(checkpoint["state_dict"])

    # Initializing the optimizer
    if opt.pretrain_path:
        opt.weight_decay = 1e-5
        opt.learning_rate = 0.001
    if opt.nesterov:
        dampening = 0
    else:
        dampening = opt.dampening

    optimizer = optim.SGD(
        parameters,
        lr=opt.learning_rate,
        momentum=opt.momentum,
        dampening=dampening,
        weight_decay=opt.weight_decay,
        nesterov=opt.nesterov,
    )

    return model, optimizer


def preprocessing_fn(inputs):
    """
    Inputs is comprised of one or more videos, where each video
    is given as an ndarray with shape (1, time, height, width, 3).
    Preprocessing resizes the height and width to 112 x 112 and reshapes
    each video to (n_stack, 3, 16, height, width), where n_stack = int(time/16).

    Outputs is a list of videos, each of shape (n_stack, 3, 16, 112, 112)
    """
    sample_duration = 16  # expected number of consecutive frames as input to the model
    outputs = []
    if inputs.dtype == np.uint8:  # inputs is a single video, i.e., batch size == 1
        inputs = [inputs]
    # else, inputs is an ndarray (of type object) of ndarrays
    for (
        input
    ) in inputs:  # each input is (1, time, height, width, 3) from the same video
        input = np.squeeze(input)

        # select a fixed number of consecutive frames
        total_frames = input.shape[0]
        if total_frames <= sample_duration:  # cyclic pad if not enough frames
            input_fixed = np.vstack(
                (input, input[: sample_duration - total_frames, ...])
            )
            assert input_fixed.shape[0] == sample_duration
        else:
            input_fixed = input

        # apply MARS preprocessing: scaling, cropping, normalizing
        opt = parse_opts(arguments=[])
        opt.modality = "RGB"
        opt.sample_size = 112
        input_Image = []  # convert each frame to PIL Image
        for f in input_fixed:
            input_Image.append(Image.fromarray(f))
        input_mars_preprocessed = preprocess_data.scale_crop(input_Image, 0, opt)

        # reshape
        input_reshaped = []
        for ns in range(int(total_frames / sample_duration)):
            np_frames = input_mars_preprocessed[
                :, ns * sample_duration : (ns + 1) * sample_duration, :, :
            ].numpy()
            input_reshaped.append(np_frames)
        outputs.append(np.array(input_reshaped, dtype=np.float32))
    return outputs


# NOTE: PyTorchClassifier expects numpy input, not torch.Tensor input
def get_art_model(model_kwargs, wrapper_kwargs, weights_file):
    model, optimizer = make_model(weights_file=weights_file, **model_kwargs)
    model.to(DEVICE)

    activity_means = np.array([114.7748, 107.7354, 99.4750])
    wrapped_model = PyTorchClassifier(
        model,
        loss=torch.nn.CrossEntropyLoss(),
        optimizer=optimizer,
        input_shape=(3, 16, 112, 112),
        nb_classes=101,
        **wrapper_kwargs,
        clip_values=(
            np.transpose(np.zeros((16, 112, 112, 3)) - activity_means, (3, 0, 1, 2)),
            np.transpose(
                255.0 * np.ones((16, 112, 112, 3)) - activity_means, (3, 0, 1, 2)
            ),
        ),
    )
    return wrapped_model
