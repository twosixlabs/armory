"""
Classifier evaluation within ARMORY
"""

import json
import os
import sys
import logging
from importlib import import_module
import numpy as np

from torch.utils.data import DataLoader
from armory.utils.config_loading import load_dataset, load_model
from armory import paths

# MARS specific imports
from MARS.opts import parse_opts
from MARS.dataset.dataset import *
from MARS.utils import AverageMeter


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


DEMO = True


def evaluate_classifier(config_path: str) -> None:
    """
    Evaluate a config file for classiifcation robustness against attack.
    """
    with open(config_path) as f:
        config = json.load(f)
    model_config = config["model"]
    classifier, preprocessing_fn = load_model(model_config)

    #################################################
    # PLACEHOLDER.  REPLACE WITH ARMORY DATAGEN
    """
    Here we use MARS native data generator for now.
    This generator assumes that all videos were first converted (off-line)
    to frames using the extract_frames.py function, whose description as follows.
    The generator outputs stacks of normalized frames, whose means = [114.7748, 107.7354, 99.4750 ]
    and stds = [1., 1., 1.].  The output shape of the generator is (nb, n_channels, n_frames, 112, 112)

    Code extracts frames from video at a rate of 25fps and scaling the
    larger dimension of the frame is scaled to 256 pixels.
    After extraction of all frames write a "done" file to signify proper completion
    of frame extraction.

    Usage:
        python extract_frames.py video_dir frame_dir

        video_dir => path of video files
        frame_dir => path of extracted jpg frames
    """

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
    opt.sample_size = 112
    opt.log = 0
    opt.batch_size = config['dataset']['batch_size']
    opt.input_channels = 3
    opt.n_workers = 0
    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
    # define location of UCF101 video frames
    opt.frame_dir=os.path.join(paths.docker().dataset_dir, "ucf101_frames/data")
    # define location of label map and data splits
    opt.annotation_path=os.path.join(paths.docker().dataset_dir, "ucf101/train_test_split_recognition_task")

    # Get model status: ucf101_trained -> fully trained; kinetics_pretrained -> needs training
    model_status = model_config['model_kwargs']['model_status']

    # NOTE: MARS datagen output has shape (nb, n_channels, n_frames, 112, 112)
    if model_status == 'kinetics_pretrained':
        print("Preprocessing train data ...")
        train_data = globals()['{}_test'.format(opt.dataset)](split = opt.split, train = 1, opt = opt)
        train_dataloader = DataLoader(train_data, batch_size = opt.batch_size, shuffle=True,
                                      num_workers = opt.n_workers, pin_memory = True, drop_last=True)
        print("Length of train data = ", len(train_data))
        print("Length of train datatloader = ",len(train_dataloader))

        print("Preprocessing validation data ...")
        val_data = globals()['{}_test'.format(opt.dataset)](split = opt.split, train = 2, opt = opt)
        print("Length of validation data = ", len(val_data))
        val_dataloader = DataLoader(val_data, batch_size = opt.batch_size, shuffle=True,
                                    num_workers = opt.n_workers, pin_memory = True, drop_last=True)
        print("Length of validation datatloader = ",len(val_dataloader))
    else: # ucf101_trained
        opt.batch_size = 1
        print("Preprocessing validation data ...")
        val_data = globals()['{}_test'.format(opt.dataset)](split = opt.split, train = 0, opt = opt)
        print("Length of validation data = ", len(val_data))
        val_dataloader = DataLoader(val_data, batch_size = opt.batch_size, shuffle=False,
                                     num_workers = opt.n_workers, pin_memory = True, drop_last=True)
        print("Length of validation datatloader = ",len(val_dataloader))
    #################################################

    # Train ART classifier
    if model_status == 'kinetics_pretrained':
        print('Train ART classifier')

    # Evaluate ART classifier on test examples
    accuracies = AverageMeter()
    for i, (clip, label) in enumerate(val_dataloader):
        clip = torch.squeeze(clip)
        x_test = np.zeros((int(clip.shape[1]/opt.sample_duration), 3, opt.sample_duration, opt.sample_size, opt.sample_size), dtype=np.float32)
        for k in range(x_test.shape[0]):
            x_test[k,:,:,:,:] = clip[:,k*opt.sample_duration:(k+1)*opt.sample_duration,:,:]
        y = classifier.predict(x_test)
        y = np.argsort(np.mean(y, axis=0))[-5:][::-1]

        acc = float(y[0] == label[0])

        accuracies.update(acc, 1)

        line = "Video[" + str(i) + "] : \t top5 " + str(y) + "\t top1 = " + str(y[0]) +  "\t true = " +str(int(label[0])) + "\t video = " + str(accuracies.avg)
        print(line)

    print("Video accuracy = ", accuracies.avg)

    '''
    logger.info(
        f"Fitting clean unpoisoned model of {model_config['module']}.{model_config['name']}..."
    )

    if DEMO:
        nb_epochs = 10
    else:
        nb_epochs = train_data_generator.total_iterations

    classifier.fit_generator(train_data_generator, nb_epochs=nb_epochs)

    # Evaluate the ART classifier on benign test examples
    logger.info("Running inference on benign examples...")
    benign_accuracy = 0
    cnt = 0

    if DEMO:
        iterations = 3
    else:
        iterations = test_data_generator.total_iterations // 2

    for _ in range(iterations):
        x, y = test_data_generator.get_batch()
        predictions = classifier.predict(x)
        benign_accuracy += np.sum(np.argmax(predictions, axis=1) == y) / len(y)
        cnt += 1
    logger.info(
        "Accuracy on benign test examples: {}%".format(benign_accuracy * 100 / cnt)
    )

    # Generate adversarial test examples
    attack_config = config["attack"]
    attack_module = import_module(attack_config["module"])
    attack_fn = getattr(attack_module, attack_config["name"])

    # Evaluate the ART classifier on adversarial test examples
    logger.info("Generating / testing adversarial examples...")

    attack = attack_fn(classifier=classifier, **attack_config["kwargs"])
    adversarial_accuracy = 0
    cnt = 0
    for _ in range(iterations):
        x, y = test_data_generator.get_batch()
        test_x_adv = attack.generate(x=x)
        predictions = classifier.predict(test_x_adv)
        adversarial_accuracy += np.sum(np.argmax(predictions, axis=1) == y) / len(y)
        cnt += 1
    logger.info(
        "Accuracy on adversarial test examples: {}%".format(
            adversarial_accuracy * 100 / cnt
        )
    )

    logger.info("Saving json output...")
    filepath = os.path.join(paths.docker().output_dir, "evaluation-results.json")
    with open(filepath, "w") as f:
        output_dict = {
            "config": config,
            "results": {
                "baseline_accuracy": str(benign_accuracy),
                "adversarial_accuracy": str(adversarial_accuracy),
            },
        }
        json.dump(output_dict, f, sort_keys=True, indent=4)
    logger.info(f"Evaluation Results written <output_dir>/evaluation-results.json")
    '''

if __name__ == "__main__":
    config_path = sys.argv[-1]
    evaluate_classifier(config_path)
