"""
Model contributed by: MITRE Corporation
Adapted from: https://github.com/craston/MARS
"""
import logging

from art.classifiers import PyTorchClassifier
import numpy as np
from PIL import Image
import torch
from torch import optim

from MARS.opts import parse_opts
from MARS.models.model import generate_model
from MARS.dataset import preprocess_data

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MEAN = np.array([114.7748, 107.7354, 99.4750], dtype=np.float32)
STD = np.array([1, 1, 1], dtype=np.float32)


def preprocessing_fn_numpy(inputs):
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


def preprocessing_fn_torch(
    batch, consecutive_frames=16, scale_first=True, align_corners=False
):
    """
    inputs - batch of videos each with shape (frames, height, width, channel)
    outputs - batch of videos each with shape (n_stack, channel, stack_frames, new_height, new_width)
        frames = n_stack * stack_frames (after padding)
        new_height = new_width = 112
    consecutive_frames - number of consecutive frames (stack_frames)

    After resizing, a center crop is performed to make the image square

    This is a differentiable alternative to MARS' PIL-based preprocessing.
        There are some
    """
    if not isinstance(batch, torch.Tensor):
        logger.warning(f"batch {type(batch)} is not a torch.Tensor. Casting")
        batch = torch.from_numpy(batch).to(DEVICE)
        # raise ValueError(f"batch {type(batch)} is not a torch.Tensor")
    if batch.dtype != torch.float32:
        raise ValueError(f"batch {batch.dtype} should be torch.float32")
    if batch.shape[0] != 1:
        raise ValueError(f"Batch size {batch.shape[0]} != 1")
    video = batch[0]

    if video.ndim != 4:
        raise ValueError(
            f"video dims {video.ndim} != 4 (frames, height, width, channel)"
        )
    if video.shape[0] < 1:
        raise ValueError("video must have at least one frame")
    if tuple(video.shape[1:]) != (240, 320, 3):
        raise ValueError(f"frame shape {tuple(video.shape[1:])} != (240, 320, 3)")
    if video.max() > 1.0 or video.min() < 0.0:
        raise ValueError("input should be float32 in [0, 1] range")
    if not isinstance(consecutive_frames, int):
        raise ValueError(f"consecutive_frames {consecutive_frames} must be an int")
    if consecutive_frames < 1:
        raise ValueError(f"consecutive_frames {consecutive_frames} must be positive")

    # Select a integer multiple of consecutive frames
    while len(video) < consecutive_frames:
        # cyclic pad if insufficient for a single stack
        video = torch.cat([video, video[: consecutive_frames - len(video)]])
    if len(video) % consecutive_frames != 0:
        # cut trailing frames
        video = video[: len(video) - (len(video) % consecutive_frames)]

    if scale_first:
        # Attempts to directly follow MARS approach
        # (frames, height, width, channel) to (frames, channel, height, width)
        video = video.permute(0, 3, 1, 2)
        sample_width, sample_height = 149, 112
        video = torch.nn.functional.interpolate(
            video,
            size=(sample_height, sample_width),
            mode="bilinear",
            align_corners=align_corners,
        )

        crop_left = 18  # round((149 - 112)/2.0)
        video = video[:, :, :, crop_left : crop_left + sample_height]

    else:
        # More efficient, but not MARS approach
        # Center crop
        sample_size = 112
        upsample, downsample = 7, 15
        assert video.shape[1] * upsample / downsample == sample_size

        crop_width = 40
        assert crop_width == (video.shape[2] - video.shape[1]) / 2
        assert video.shape[1] + 2 * crop_width == video.shape[2]

        video = video[:, :, crop_width : video.shape[2] - crop_width, :]
        assert video.shape[1] == video.shape[2] == 240

        # Downsample to (112, 112) frame size
        # (frames, height, width, channel) to (frames, channel, height, width)
        video = video.permute(0, 3, 1, 2)
        video = torch.nn.functional.interpolate(
            video,
            size=(sample_size, sample_size),
            mode="bilinear",
            align_corners=align_corners,
        )

    if video.max() > 1.0:
        raise ValueError("Video exceeded max after interpolation")
    if video.min() < 0.0:
        raise ValueError("Video under min after interpolation")

    # reshape into stacks of frames
    video = torch.reshape(video, (-1, consecutive_frames) + video.shape[1:])

    # transpose to (stacks, channel, stack_frames, height, width)
    video = video.permute(0, 2, 1, 3, 4)
    # video = torch.transpose(video, axes=(0, 4, 1, 2, 3))

    # normalize before changing channel position?
    video = torch.transpose(video, 1, 4)
    video = ((video * 255) - torch.from_numpy(MEAN).to(DEVICE)) / torch.from_numpy(
        STD
    ).to(DEVICE)
    video = torch.transpose(video, 4, 1)

    return video


def fit_preprocessing_fn_numpy(batch):
    """
    Randomly sample a single stack from each video
    """
    x = preprocessing_fn_numpy(batch)
    x = np.stack([x_i[np.random.randint(x_i.shape[0])] for x_i in x])
    return x


preprocessing_fn = fit_preprocessing_fn_numpy


def make_model(model_status="ucf101_trained", weights_path=None):
    statuses = ("ucf101_trained", "kinetics_pretrained")
    if model_status not in statuses:
        raise ValueError(f"model_status {model_status} not in {statuses}")
    trained = model_status == "ucf101_trained"
    if not trained and weights_path is None:
        raise ValueError("weights_path cannot be None for 'kinetics_pretrained'")

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

        opt.pretrain_path = weights_path

    logger.info(f"Loading model... {opt.model} {opt.model_depth}")
    model, parameters = generate_model(opt)

    if trained and weights_path is not None:
        checkpoint = torch.load(weights_path, map_location=DEVICE)
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


class OuterModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        if self.training:
            # Use preprocessing_fn_numpy in dataset preprocessing
            return self.model(x)
        else:
            x = preprocessing_fn_torch(x)
            stack_outputs = self.model(x)
            output = stack_outputs.mean(axis=0, keepdims=True)

        return output


def get_art_model(model_kwargs, wrapper_kwargs, weights_path):
    inner_model, optimizer = make_model(weights_path=weights_path, **model_kwargs)
    inner_model.to(DEVICE)
    model = OuterModel(inner_model)
    model.to(DEVICE)

    wrapped_model = PyTorchClassifier(
        model,
        loss=torch.nn.CrossEntropyLoss(),
        optimizer=optimizer,
        input_shape=(None, 240, 320, 3),
        nb_classes=101,
        clip_values=(0.0, 1.0),
        **wrapper_kwargs,
    )
    return wrapped_model
