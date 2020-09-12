import logging

logger = logging.getLogger(__name__)

from art.estimation.classification import PyTorchClassifier
import numpy as np
import torch
from torch import optim

from MARS.opts import parse_opts
from MARS.models.model import generate_model

# from MARS.dataset import preprocess_data

from armory.data.utils import maybe_download_weights_from_s3

# from armory.data import datasets
from armory.scenarios.video_update import dataset_canonical_preprocessing
from armory.baseline_models.pytorch.ucf101_mars import preprocessing_fn as orig_fn

MEAN = np.array([114.7748, 107.7354, 99.4750], dtype=np.float32)
STD = np.array([1, 1, 1], dtype=np.float32)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocessing_torch(
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
        batch = torch.from_numpy(batch)
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
    if video.min() > 1.0:
        raise ValueError("Video under min after interpolation")

    # reshape into stacks of frames
    video = torch.reshape(video, (-1, consecutive_frames) + video.shape[1:])

    # transpose to (stacks, channel, stack_frames, height, width)
    video = video.permute(0, 2, 1, 3, 4)
    # video = torch.transpose(video, axes=(0, 4, 1, 2, 3))

    # normalize before changing channel position?
    video = torch.transpose(video, 1, 4)
    video = ((video * 255) - torch.from_numpy(MEAN)) / torch.from_numpy(STD)
    video = torch.transpose(video, 4, 1)

    return video


def process_both(x):
    original = orig_fn(x)
    original = original[0]
    numpy_canon = dataset_canonical_preprocessing(x)
    update = torch.from_numpy(numpy_canon)
    update = preprocessing_torch(update)
    numpy = update.numpy()

    outputs = []
    for y in original, numpy:
        y = y.transpose(0, 2, 3, 4, 1)
        y = y + MEAN
        outputs.append(y)
    return outputs[0], outputs[1]


def compare_videos(a, b):
    a = a.round().astype(np.uint8)
    b = b.round().astype(np.uint8)

    rows = []
    for i, (ai, bi) in enumerate(zip(a, b)):
        row = []
        for j, (aj, bj) in enumerate(zip(ai, bi)):
            cj = np.vstack([aj, bj])
            row.append(cj)
        rows.append(np.hstack(row))
    rows = np.vstack(rows)
    return rows


def save_video(c, name):
    from PIL import Image

    c = c.round().astype(np.uint8)
    rows = []
    for ci in c:
        row = np.hstack(ci)
        rows.append(row)
    rows = np.vstack(rows)
    Image.fromarray(rows).convert("RGB").save(name)
    return rows


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


class OuterModel(torch.nn.Module):
    def __init__(self, model):
        self.model = model

    def forward(self, x):
        x = preprocessing_torch(x)

        if self.training:
            raise NotImplementedError("training mode not complete yet")
            # self.model(x)
        else:
            stack_outputs = self.model(x)
            output = stack_outputs.mean(axis=0)

        return output


def get_art_model(model_kwargs, wrapper_kwargs, weights_file):
    inner_model, optimizer = make_model(weights_file=weights_file, **model_kwargs)
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
