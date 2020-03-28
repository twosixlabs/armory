"""
Classifier evaluation within ARMORY
"""

import json
import os
import sys
import logging
from importlib import import_module
import numpy as np
import time

from torch.utils.data import DataLoader
from armory.utils.config_loading import load_dataset, load_model
from armory import paths

# MARS specific imports
from MARS.opts import parse_opts
from MARS.dataset.dataset import *
from MARS.utils import AverageMeter

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def evaluate_classifier(config_path: str) -> None:
    """
    Evaluate a config file for classiifcation robustness against attack.
    """
    with open(config_path) as f:
        config = json.load(f)
    model_config = config["model"]
    # Get model status: ucf101_trained -> fully trained; kinetics_pretrained -> needs training
    model_status = model_config['model_kwargs']['model_status']
    classifier, preprocessing_fn = load_model(model_config)

    '''
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
    '''

    #################################################
    # Armory UCF101 Datagen
    logger.info(f"Loading dataset {config['dataset']['name']}...")
    train_epochs = config["adhoc"]["train_epochs"]
    train_data_generator = load_dataset(
        config["dataset"],
        epochs=train_epochs,
        split_type="train",
        preprocessing_fn=preprocessing_fn,
    )

    # Train ART classifier
    if model_status == 'kinetics_pretrained':
        logger.info(f"Fitting clean model of {model_config['module']}.{model_config['name']}...")
        logger.info(f"Loading training dataset {config['dataset']['name']}...")
        train_epochs = config["adhoc"]["train_epochs"]
        batch_size = config['dataset']['batch_size']
        train_data_generator = load_dataset(
            config["dataset"],
            epochs=train_epochs,
            split_type="train",
            preprocessing_fn=preprocessing_fn,
        )

        for e in range(train_epochs):
            classifier.set_learning_phase(True)
            st_time = time.time()
            for b in range(train_data_generator.batches_per_epoch):
                logger.info(f"Epoch: {e}/{train_epochs}, batch: {b}/{train_data_generator.batches_per_epoch}")
                x_trains, y_trains = train_data_generator.get_batch()
                # x_trains consists of one or more videos, each represented as a ndarray of shape
                # (n_stacks, 3, 16, 112, 112).  To train, randomly sample a batch of stacks
                x_train = np.zeros((min(batch_size, len(x_trains)), 3, 16, 112, 112), dtype=np.float32)
                for i,xt in enumerate(x_trains):
                    rand_stack = np.random.randint(0,xt.shape[0])
                    x_train[i,...] = xt[rand_stack,...]
                classifier.fit(x_train, y_trains, batch_size=batch_size, nb_epochs=1)
            logger.info("Time per epoch: {}s".format(time.time()-st_time))

            # evaluate on test examples
            classifier.set_learning_phase(False)
            test_data_generator = load_dataset(
                config["dataset"],
                epochs=1,
                split_type="test",
                preprocessing_fn=preprocessing_fn,
            )

            accuracies = AverageMeter()
            video_count = 0
            for i in range(int(test_data_generator.batches_per_epoch/10)):
                x_tests, y_tests = test_data_generator.get_batch()
                for x_test, y_test in zip(x_tests, y_tests): # each x_test is of shape (n_stack, 3, 16, 112, 112) and represents a video
                    y = classifier.predict(x_test)
                    y = np.argsort(np.mean(y, axis=0))[-5:][::-1]
                    acc = float(y[0] == y_test)
                    accuracies.update(acc, 1)
            logger.info("Video accuracy = {}".format(accuracies.avg))

    # Evaluate ART classifier on test examples
    logger.info("Running inference on benign test examples...")
    logger.info(f"Loading testing dataset {config['dataset']['name']}...")
    classifier.set_learning_phase(False)
    test_data_generator = load_dataset(
        config["dataset"],
        epochs=1,
        split_type="test",
        preprocessing_fn=preprocessing_fn,
    )

    accuracies = AverageMeter()
    video_count = 0
    for i in range(test_data_generator.batches_per_epoch):
        x_tests, y_tests = test_data_generator.get_batch()
        for x_test, y_test in zip(x_tests, y_tests): # each x_test is of shape (n_stack, 3, 16, 112, 112) and represents a video
            y = classifier.predict(x_test)
            y = np.argsort(np.mean(y, axis=0))[-5:][::-1]
            acc = float(y[0] == y_test)

            accuracies.update(acc, 1)

            line = "Video[" + str(video_count) + "] : \t top5 " + str(y) + "\t top1 = " + str(y[0]) +  "\t true = " +str(y_test) + "\t video_acc = " + str(accuracies.avg)
            logger.info(line)
            video_count += 1

    print("Video accuracy = ", accuracies.avg)

    '''
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
