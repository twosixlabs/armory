"""
Classifier evaluation within ARMORY
"""

import json
import os
import sys
import logging
import coloredlogs
from importlib import import_module
import time

import numpy as np

from armory import paths
from armory.utils.metrics import AverageMeter
from armory.utils.config_loading import load_dataset, load_model


logger = logging.getLogger(__name__)
coloredlogs.install(logging.DEBUG)


def evaluate_classifier(config_path: str) -> None:
    """
    Evaluate a config file for classification robustness against attack.
    """
    with open(config_path) as f:
        config = json.load(f)
    model_config = config["model"]
    # Get model status: ucf101_trained -> fully trained; kinetics_pretrained -> needs training
    model_status = model_config["model_kwargs"]["model_status"]
    classifier, preprocessing_fn = load_model(model_config)

    # Train ART classifier
    # Armory UCF101 Datagen
    logger.info(f"Loading dataset {config['dataset']['name']}...")
    train_epochs = config["adhoc"]["train_epochs"]
    train_data_generator = load_dataset(
        config["dataset"],
        epochs=train_epochs,
        split_type="train",
        preprocessing_fn=preprocessing_fn,
    )

    if model_status == "kinetics_pretrained":
        logger.info(
            f"Fitting clean model of {model_config['module']}.{model_config['name']}..."
        )
        logger.info(f"Loading training dataset {config['dataset']['name']}...")
        train_epochs = config["adhoc"]["train_epochs"]
        batch_size = config["dataset"]["batch_size"]
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
                logger.info(
                    f"Epoch: {e}/{train_epochs}, batch: {b}/{train_data_generator.batches_per_epoch}"
                )
                x_trains, y_trains = train_data_generator.get_batch()
                # x_trains consists of one or more videos, each represented as a ndarray of shape
                # (n_stacks, 3, 16, 112, 112).  To train, randomly sample a batch of stacks
                x_train = np.zeros(
                    (min(batch_size, len(x_trains)), 3, 16, 112, 112), dtype=np.float32
                )
                for i, xt in enumerate(x_trains):
                    rand_stack = np.random.randint(0, xt.shape[0])
                    x_train[i, ...] = xt[rand_stack, ...]
                classifier.fit(x_train, y_trains, batch_size=batch_size, nb_epochs=1)
            logger.info("Time per epoch: {}s".format(time.time() - st_time))

            # evaluate on test examples
            classifier.set_learning_phase(False)
            test_data_generator = load_dataset(
                config["dataset"],
                epochs=1,
                split_type="test",
                preprocessing_fn=preprocessing_fn,
            )

            accuracies = AverageMeter()
            accuracies_top5 = AverageMeter()
            video_count = 0
            for i in range(int(test_data_generator.batches_per_epoch // 10)):
                x_tests, y_tests = test_data_generator.get_batch()
                for x_test, y_test in zip(
                    x_tests, y_tests
                ):  # each x_test is of shape (n_stack, 3, 16, 112, 112) and represents a video
                    y = classifier.predict(x_test)
                    y = np.argsort(np.mean(y, axis=0))[-5:][::-1]
                    acc = float(y[0] == y_test)
                    acc_top5 = float(y_test in y)
                    accuracies.update(acc, 1)
                    accuracies_top5.update(acc_top5, 1)
            logger.info(
                "Top-1 video accuracy = {}, top-5 video accuracy = {}".format(
                    accuracies.avg, accuracies_top5.avg
                )
            )

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

    test_accuracies = AverageMeter()
    test_accuracies_top5 = AverageMeter()
    video_count = 0
    for i in range(test_data_generator.batches_per_epoch):
        x_tests, y_tests = test_data_generator.get_batch()
        for x_test, y_test in zip(
            x_tests, y_tests
        ):  # each x_test is of shape (n_stack, 3, 16, 112, 112) and represents a video
            y = classifier.predict(x_test)
            y = np.argsort(np.mean(y, axis=0))[-5:][::-1]
            acc = float(y[0] == y_test)
            acc_top5 = float(y_test in y)

            test_accuracies.update(acc, 1)
            test_accuracies_top5.update(acc_top5, 1)

            logger.info(
                "\t ".join(
                    [
                        f"Video[{video_count}] : ",
                        f"top5 = {y}",
                        f"top1 = {y[0]}",
                        f"true = {y_test}",
                        f"top1_video_acc = {test_accuracies.avg}",
                        f"top5_video_acc = {test_accuracies_top5.avg}",
                    ]
                )
            )
            video_count += 1

    logger.info(
        "Top-1 test video accuracy = {}, top-5 test video accuracy = {}".format(
            test_accuracies.avg, test_accuracies_top5.avg
        )
    )

    # Evaluate the ART classifier on adversarial test examples
    logger.info("Generating / testing adversarial examples...")

    # Generate adversarial test examples
    attack_config = config["attack"]
    attack_module = import_module(attack_config["module"])
    attack_fn = getattr(attack_module, attack_config["name"])

    classifier.set_learning_phase(False)
    test_data_generator = load_dataset(
        config["dataset"],
        epochs=1,
        split_type="test",
        preprocessing_fn=preprocessing_fn,
    )

    adv_accuracies = AverageMeter()
    adv_accuracies_top5 = AverageMeter()
    video_count = 0
    for i in range(test_data_generator.batches_per_epoch // 10):
        x_tests, y_tests = test_data_generator.get_batch()
        for x_test, y_test in zip(x_tests, y_tests):
            # each x_test is of shape (n_stack, 3, 16, 112, 112) and represents a video
            attack = attack_fn(
                classifier=classifier,
                **attack_config["kwargs"],
                batch_size=x_test.shape[0],
            )
            test_x_adv = attack.generate(x=x_test)
            y = classifier.predict(test_x_adv)
            y = np.argsort(np.mean(y, axis=0))[-5:][::-1]
            acc = float(y[0] == y_test)
            acc_top5 = float(y_test in y)

            adv_accuracies.update(acc, 1)
            adv_accuracies_top5.update(acc_top5, 1)

            logger.info(
                "\t ".join(
                    [
                        f"Video[{video_count}] : ",
                        f"top5 = {y}",
                        f"top1 = {y[0]}",
                        f"true = {y_test}",
                        f"top1_video_acc = {adv_accuracies.avg}",
                        f"top5_video_acc = {adv_accuracies_top5.avg}",
                    ]
                )
            )
            video_count += 1

    logger.info(
        "Top-1 adversarial video accuracy = {}, top-5 adversarial video accuracy = {}".format(
            adv_accuracies.avg, adv_accuracies_top5.avg
        )
    )

    logger.info("Saving json output...")
    filepath = os.path.join(paths.docker().output_dir, "ucf101_evaluation-results.json")
    with open(filepath, "w") as f:
        output_dict = {
            "config": config,
            "results": {
                "baseline_top1_accuracy": str(test_accuracies.avg),
                "baseline_top5_accuracy": str(test_accuracies_top5.avg),
                "adversarial_top1_accuracy": str(adv_accuracies.avg),
                "adversarial_top5_accuracy": str(adv_accuracies_top5.avg),
            },
        }
        json.dump(output_dict, f, sort_keys=True, indent=4)
    logger.info(
        f"Evaluation Results written to {paths.docker().output_dir}/ucf101_evaluation-results.json"
    )


if __name__ == "__main__":
    config_path = sys.argv[-1]
    evaluate_classifier(config_path)
