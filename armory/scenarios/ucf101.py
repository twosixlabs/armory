"""
Classifier evaluation within ARMORY

Scenario Contribuor: MITRE Corporation
"""

import logging

import numpy as np
from tqdm import tqdm

from armory.scenarios.base import Scenario
from armory.utils.metrics import AverageMeter
from armory.utils.config_loading import load_dataset, load_model, load_attack

logger = logging.getLogger(__name__)


def update_accuracies(y_pred, y, accuracies, accuracies_top5):
    """
    Update top-1 and top-5 accuracies. Return top-5 examples, sorted.
    """
    y_pred_top5 = np.argsort(np.mean(y_pred, axis=0))[-5:][::-1]
    acc = float(y_pred_top5[0] == y)
    acc_top5 = float(y in y_pred_top5)
    accuracies.update(acc, 1)
    accuracies_top5.update(acc_top5, 1)
    return y_pred_top5


class Ucf101(Scenario):
    def _evaluate(self, config: dict) -> dict:
        """
        Evaluate the config and return a results dict
        """
        model_config = config["model"]
        classifier, preprocessing_fn = load_model(model_config)

        if model_config.get("fit"):
            logger.info(
                f"Fitting model of {model_config['module']}.{model_config['name']}..."
            )
            logger.info(f"Loading training dataset {config['dataset']['name']}...")
            train_epochs = config["model"]["fit_kwargs"]["nb_epochs"]
            batch_size = config["dataset"]["batch_size"]
            train_data_generator = load_dataset(
                config["dataset"],
                epochs=train_epochs,
                split_type="train",
                preprocessing_fn=preprocessing_fn,
            )

            for epoch in range(train_epochs):
                classifier.set_learning_phase(True)

                for _ in tqdm(
                    range(train_data_generator.batches_per_epoch),
                    desc=f"Epoch: {epoch}/{train_epochs}",
                ):
                    x_trains, y_trains = train_data_generator.get_batch()
                    # x_trains consists of one or more videos, each represented as an
                    # ndarray of shape (n_stacks, 3, 16, 112, 112).
                    # To train, randomly sample a batch of stacks
                    x_train = np.zeros(
                        (min(batch_size, len(x_trains)), 3, 16, 112, 112),
                        dtype=np.float32,
                    )
                    for i, xt in enumerate(x_trains):
                        rand_stack = np.random.randint(0, xt.shape[0])
                        x_train[i, ...] = xt[rand_stack, ...]
                    classifier.fit(
                        x_train, y_trains, batch_size=batch_size, nb_epochs=1
                    )

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
                for _ in tqdm(range(len(test_data_generator) // 10), desc="Validation"):
                    x_tests, y_tests = test_data_generator.get_batch()
                    for x_test, y_test in zip(x_tests, y_tests):
                        y_pred = classifier.predict(x_test)
                        update_accuracies(y_pred, y_test, accuracies, accuracies_top5)
                logger.info(
                    f"Top-1 video accuracy = {accuracies.avg}, "
                    f"top-5 video accuracy = {accuracies_top5.avg}"
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
        for x_tests, y_tests in tqdm(test_data_generator, desc="Benign"):
            for x_test, y_test in zip(x_tests, y_tests):
                y_pred = classifier.predict(x_test)
                y_pred_top5 = update_accuracies(
                    y_pred, y_test, accuracies, accuracies_top5
                )

                logger.info(
                    "\t ".join(
                        [
                            f"Video[{video_count}] : ",
                            f"top5 = {y_pred_top5}",
                            f"top1 = {y_pred_top5[0]}",
                            f"true = {y_test}",
                            f"top1_video_acc = {test_accuracies.avg}",
                            f"top5_video_acc = {test_accuracies_top5.avg}",
                        ]
                    )
                )
                video_count += 1

        logger.info(
            f"Top-1 test video accuracy = {test_accuracies.avg}, "
            f"top-5 test video accuracy = {test_accuracies_top5.avg}"
        )

        # Evaluate the ART classifier on adversarial test examples
        logger.info("Generating / testing adversarial examples...")
        attack = load_attack(config["attack"], classifier)

        test_data_generator = load_dataset(
            config["dataset"],
            epochs=1,
            split_type="test",
            preprocessing_fn=preprocessing_fn,
        )

        adv_accuracies = AverageMeter()
        adv_accuracies_top5 = AverageMeter()
        video_count = 0
        for _ in tqdm(range(len(test_data_generator)), desc="Attack"):
            x_tests, y_tests = test_data_generator.get_batch()
            for x_test, y_test in zip(x_tests, y_tests):
                # each x_test is of shape (n_stack, 3, 16, 112, 112)
                #    n_stack varies
                attack.set_params(batch_size=x_test.shape[0])
                test_x_adv = attack.generate(x=x_test)
                y_pred = classifier.predict(test_x_adv)
                y_pred_top5 = update_accuracies(
                    y_pred, y_test, accuracies, accuracies_top5
                )

                logger.info(
                    "\t ".join(
                        [
                            f"Video[{video_count}] : ",
                            f"top5 = {y_pred_top5}",
                            f"top1 = {y_pred_top5[0]}",
                            f"true = {y_test}",
                            f"top1_video_acc = {test_accuracies.avg}",
                            f"top5_video_acc = {test_accuracies_top5.avg}",
                        ]
                    )
                )

                video_count += 1

        logger.info(
            f"Top-1 adversarial video accuracy = {adv_accuracies.avg}, "
            f"top-5 adversarial video accuracy = {adv_accuracies_top5.avg}"
        )
        return {
            "baseline_top1_accuracy": str(test_accuracies.avg),
            "baseline_top5_accuracy": str(test_accuracies_top5.avg),
            "adversarial_top1_accuracy": str(adv_accuracies.avg),
            "adversarial_top5_accuracy": str(adv_accuracies_top5.avg),
        }
