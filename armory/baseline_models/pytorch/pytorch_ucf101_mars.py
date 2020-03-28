import logging
import os
import sys

from art.classifiers import PyTorchClassifier
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch import optim
from torchvision import models
from torch.optim import lr_scheduler
from armory import paths

# MARS specific imports
from MARS.opts import parse_opts
from MARS.models.model import generate_model
from MARS.dataset import preprocess_data

#logger = logging.getLogger(__name__)
#os.environ["TORCH_HOME"] = os.path.join(paths.docker().dataset_dir, "pytorch", "models")


def make_model(**kwargs):
    sys.argv=[''];
    opt = parse_opts()

    # Default opts for UCF101 dataset
    opt.dataset = 'UCF101'
    opt.modality = 'RGB'
    opt.split = 1
    opt.only_RGB = True
    opt.model = 'resnext'
    opt.model_depth = 101
    opt.sample_duration = 16
    opt.log = 0
    opt.batch_size = 1
    opt.input_channels = 3
    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)

    model_status = kwargs.get('model_status', 'kinetics_pretrained')
    if model_status == 'ucf101_trained':
        opt.n_classes = 101
        opt.resume_path1 = os.path.join(paths.docker().saved_model_dir, "mars", "MARS_UCF101_16f.pth")
    else:
        opt.n_classes = 400
        opt.pretrain_path = os.path.join(paths.docker().saved_model_dir, "mars", "RGB_Kinetics_16f.pth")
        opt.n_finetune_classes = 101
        opt.batch_size = 32
        opt.ft_begin_index = 4

    print("Loading model... ", opt.model, opt.model_depth)
    model, parameters = generate_model(opt)

    # Loading trained model weights
    if opt.resume_path1:
        print('loading checkpoint for UCF101 trained model {}'.format(opt.resume_path1))
        checkpoint = torch.load(opt.resume_path1)
        assert opt.arch == checkpoint['arch']
        opt.begin_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])

    # Initializing the optimizer
    if opt.pretrain_path:
        opt.weight_decay = 1e-5
        opt.learning_rate = 0.001
    if opt.nesterov: dampening = 0
    else: dampening = opt.dampening

    print("lr = {} \t momentum = {} \t dampening = {} \t weight_decay = {}, \t nesterov = {}"
                .format(opt.learning_rate, opt.momentum, dampening, opt. weight_decay, opt.nesterov))
    print("LR patience = ", opt.lr_patience)

    optimizer = optim.SGD(
        parameters,
        lr=opt.learning_rate,
        momentum=opt.momentum,
        dampening=dampening,
        weight_decay=opt.weight_decay,
        nesterov=opt.nesterov)

    return model, optimizer

def preprocessing_fn(inputs):
    """
    Inputs is comprised of one or more videos, where each video
    is given as an ndarray with shape (1, time, height, width, 3).
    Preprocessing resizes the height and width to 112 x 112 and reshapes
    each video to (n_stack, 3, 16, height, width), where n_stack = int(time/16).

    Outputs is a list of videos, each of shape (n_stack, 3, 16, 112, 112)
    """
    sample_duration = 16 # expected number of consecutive frames as input to the model
    outputs = []
    if inputs.dtype == np.uint8: # inputs is a single video, i.e., batch size == 1
        inputs = [inputs]
    # else, inputs is an ndarray (of type object) of ndarrays
    for input in inputs: # each input is (1, time, height, width, 3) from the same video
        input = np.squeeze(input)

        # select a fixed number of consecutive frames
        total_frames = input.shape[0]
        if total_frames <= sample_duration: # cyclic pad if not enough frames
            input_fixed = np.vstack((input, input[:sample_duration-total_frames,...]))
            assert(input_fixed.shape[0] == sample_duration)
        else: input_fixed = input

        # apply MARS preprocessing: scaling, cropping, normalizing
        opt = parse_opts()
        opt.modality = 'RGB'
        opt.sample_size = 112
        input_Image = [] # convert each frame to PIL Image
        for f in input_fixed:
            input_Image.append(Image.fromarray(f))
        input_mars_preprocessed = preprocess_data.scale_crop(input_Image, 0, opt)

        # reshape
        input_reshaped = []
        for ns in range(int(total_frames/sample_duration)):
            np_frames = input_mars_preprocessed[:,ns*sample_duration:(ns+1)*sample_duration,:,:].numpy()
            input_reshaped.append(np_frames)
        outputs.append(np.array(input_reshaped, dtype=np.float32))
    return outputs

# NOTE: PyTorchClassifier expects numpy input, not torch.Tensor input
def get_art_model(model_kwargs, wrapper_kwargs):
    model, optimizer = make_model(**model_kwargs)

    UCF101_MEANS = [114.7748, 107.7354, 99.4750]
    UCF101_STDEV = [1, 1, 1]

    wrapped_model = PyTorchClassifier(
        model,
        loss=torch.nn.CrossEntropyLoss().cuda(),
        optimizer=optimizer,
        input_shape=(3, 16, 112, 112),
        nb_classes = 101,
        **wrapper_kwargs,
        clip_values=(
            np.array(
                [
                    (0.0 - UCF101_MEANS[0]) / UCF101_STDEV[0],
                    (0.0 - UCF101_MEANS[1]) / UCF101_STDEV[1],
                    (0.0 - UCF101_MEANS[2]) / UCF101_STDEV[2],
                ]
            ),
            np.array(
                [
                    (255.0 - UCF101_MEANS[0]) / UCF101_STDEV[0],
                    (255.0 - UCF101_MEANS[1]) / UCF101_STDEV[1],
                    (255.0 - UCF101_MEANS[2]) / UCF101_STDEV[2],
                ]
            ),
        )
    )
    return wrapped_model
