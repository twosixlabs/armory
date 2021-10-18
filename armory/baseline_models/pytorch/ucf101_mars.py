"""
Model contributed by: MITRE Corporation
Adapted from: https://github.com/craston/MARS
"""
import logging
from typing import Union, Optional, Tuple

from art.estimators.classification import PyTorchClassifier
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


def preprocessing_fn_numpy(batch: np.ndarray):
    """
    batch is a batch of videos, (batch, frames, height, width, channels)

    Preprocessing resizes the height and width to 112 x 112 and reshapes
    each video to (n_stack, 3, 16, height, width), where n_stack = int(time/16).

    Outputs a list of videos, each of shape (n_stack, 3, 16, 112, 112)
    """
    sample_duration = 16  # expected number of consecutive frames as input to the model

    outputs = []
    for i, x in enumerate(batch):
        if x.ndim != 4:
            raise ValueError(f"sample {i} in batch has {x.ndim} dims, not 4 (FHWC)")
        if x.dtype in (float, np.float32):
            if x.max() > 1.0 or x.min() < 0.0:
                raise ValueError(f"sample {i} is a float but not in [0.0, 1.0] range")
            x = (255 * x).round().astype(np.uint8)
        if x.dtype != np.uint8:
            raise ValueError(f"sample {i} - unrecognized dtype {x.dtype}")

        # select a fixed number of consecutive frames
        total_frames = x.shape[0]
        if total_frames < sample_duration / 2:
            raise ValueError(
                f"video is too short; requires >= {sample_duration / 2} frames"
            )
        if total_frames <= sample_duration:  # cyclic pad if not enough frames
            x = np.vstack([x, x[: sample_duration - total_frames]])

        # apply MARS preprocessing: scaling, cropping, normalizing
        opt = parse_opts(arguments=[])
        opt.modality = "RGB"
        opt.sample_size = 112
        x_Image = []  # convert each frame to PIL Image
        for frame in x:
            x_Image.append(Image.fromarray(frame))
        x_mars_preprocessed = preprocess_data.scale_crop(x_Image, 0, opt)

        # reshape
        x_reshaped = []
        for ns in range(int(total_frames / sample_duration)):
            np_frames = x_mars_preprocessed[
                :, ns * sample_duration : (ns + 1) * sample_duration, :, :
            ].numpy()
            x_reshaped.append(np_frames)
        outputs.append(np.array(x_reshaped, dtype=np.float32))
    return outputs


def preprocessing_fn_torch(
    batch: Union[torch.Tensor, np.ndarray],
    consecutive_frames: int = 16,
    scale_first: bool = True,
    align_corners: bool = False,
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
    if tuple(video.shape[1:]) == (240, 320, 3):
        standard_shape = True
    elif tuple(video.shape[1:]) == (226, 400, 3):
        logger.warning("Expected odd example shape (226, 400, 3)")
        standard_shape = False
    else:
        raise ValueError(f"frame shape {tuple(video.shape[1:])} not recognized")
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
        if standard_shape:  # 240 x 320
            sample_height, sample_width = 112, 149
        else:  # 226 x 400
            video = video[:, :, 1:-1, :]  # crop top/bottom pixels, reduce by 2
            sample_height, sample_width = 112, 200

        video = torch.nn.functional.interpolate(
            video,
            size=(sample_height, sample_width),
            mode="bilinear",
            align_corners=align_corners,
        )

        if standard_shape:
            crop_left = 18  # round((149 - 112)/2.0)
        else:
            crop_left = 40
        video = video[:, :, :, crop_left : crop_left + sample_height]

    else:
        # More efficient, but not MARS approach
        # Center crop
        sample_size = 112
        if standard_shape:
            crop_width = 40
        else:
            video = video[:, 1:-1, :, :]
            crop_width = 88
        video = video[:, :, crop_width:-crop_width, :]

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


def fit_preprocessing_fn_numpy(batch: np.ndarray):
    """
    Randomly sample a single stack from each video
    """
    x = preprocessing_fn_numpy(batch)
    x = np.stack([x_i[np.random.randint(x_i.shape[0])] for x_i in x])
    return x


preprocessing_fn = fit_preprocessing_fn_numpy


def make_model(
    model_status: str = "ucf101_trained", weights_path: Optional[str] = None
) -> Tuple[torch.nn.DataParallel, optim.SGD]:
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
    def __init__(
        self, weights_path: Optional[str], max_frames: int = 0, **model_kwargs,
    ):
        """
        Max frames is the maximum number of input frames.
            If max_frames == 0, no clipping is done
            Else if max_frames > 0, frames are clipped to that number.
            This can be helpful for smaller memory cards.
        """
        super().__init__()
        max_frames = int(max_frames)
        if max_frames < 0:
            raise ValueError(f"max_frames {max_frames} cannot be negative")
        self.max_frames = max_frames
        self.model, self.optimizer = make_model(
            weights_path=weights_path, **model_kwargs
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.max_frames:
            x = x[:, : self.max_frames]

        if self.training:
            # Use preprocessing_fn_numpy in dataset preprocessing
            return self.model(x)
        else:
            x = preprocessing_fn_torch(x)
            stack_outputs = self.model(x)
            output = stack_outputs.mean(axis=0, keepdims=True)

        return output


def get_art_model(
    model_kwargs: dict, wrapper_kwargs: dict, weights_path: Optional[str] = None
) -> PyTorchClassifier:
    model = OuterModel(weights_path=weights_path, **model_kwargs)
    model.to(DEVICE)

    wrapped_model = PyTorchClassifier(
        model,
        loss=torch.nn.CrossEntropyLoss(),
        optimizer=model.optimizer,
        input_shape=(None, 240, 320, 3),
        channels_first=False,
        nb_classes=101,
        clip_values=(0.0, 1.0),
        **wrapper_kwargs,
    )
    return wrapped_model
