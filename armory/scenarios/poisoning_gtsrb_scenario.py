"""
Classifier evaluation within ARMORY

Scenario Contributor: MITRE Corporation
"""

import logging

import numpy as np
from tensorflow import set_random_seed
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

from armory.utils.config_loading import (
    load_dataset,
    load_model,
    load,
    load_fn,
)
from armory.utils import metrics
from armory.scenarios.base import Scenario

logger = logging.getLogger(__name__)


def poison_batch(src_imgs, src_lbls, src, tgt, batch_size, attack):
    # In this example, all images of "src" class have a trigger
    # added and re-labeled as "tgt" class
    # NOTE: currently art.attacks.PoisonAttackBackdoor only supports
    #   black-white images.  One way to generate poisoned examples
    #   is to convert each batch of multi-channel images of shape
    #   (N,W,H,C) to N separate (C,W,H)-tuple, where C would be
    #   interpreted by PoisonAttackBackdoor as the batch size,
    #   and each channel would have a backdoor trigger added
    poison_x = []
    poison_y = []
    for idx in range(batch_size):
        if src_lbls[idx] == src:
            src_img = np.transpose(src_imgs[idx], (2, 0, 1))
            p_img, p_label = attack.poison(src_img, [tgt])
            poison_x.append(np.transpose(p_img, (1, 2, 0)))
            poison_y.append(p_label)
        else:
            poison_x.append(src_imgs[idx])
            poison_y.append(src_lbls[idx])

    poison_x, poison_y = np.array(poison_x), np.array(poison_y)

    return poison_x, poison_y


class GTSRB(Scenario):
    def _evaluate(self, config: dict) -> dict:
        """
        Evaluate a config file for classification robustness against attack.
        """

        model_config = config["model"]
        classifier, preprocessing_fn = load_model(model_config)
        classifier_for_defense, _ = load_model(model_config)

        train_epochs = config["adhoc"]["train_epochs"]
        src_class = config["adhoc"]["source_class"]
        tgt_class = config["adhoc"]["target_class"]

        # Set random seed due to large variance in attack and defense success
        np.random.seed(config["adhoc"]["np_seed"])
        set_random_seed(config["adhoc"]["tf_seed"])

        logger.info(f"Loading dataset {config['dataset']['name']}...")
        batch_size = config["dataset"]["batch_size"]
        train_data = load_dataset(
            config["dataset"],
            epochs=1,
            split_type="train",
            preprocessing_fn=preprocessing_fn,
        )

        logger.info("Building in-memory dataset for poisoning detection and training")
        attack_config = config["attack"]
        attack = load(attack_config)
        fraction_poisoned = config["adhoc"]["fraction_poisoned"]
        poison_dataset_flag = config["adhoc"]["poison_dataset"]
        # detect_poison does not currently support data generators
        #     therefore, make in memory dataset
        x_train_all, y_train_all = [], []
        for x_train, y_train in train_data:
            if poison_dataset_flag and np.random.rand() < fraction_poisoned:
                x_train, y_train = poison_batch(
                    x_train, y_train, src_class, tgt_class, len(y_train), attack
                )
            x_train_all.append(x_train)
            y_train_all.append(y_train)
        x_train_all = np.concatenate(x_train_all, axis=0)
        y_train_all = np.concatenate(y_train_all, axis=0)
        y_train_all_categorical = to_categorical(y_train_all)

        defense_config = config["defense"]
        logger.info(
            f"Fitting model {model_config['module']}.{model_config['name']} "
            f"for defense {defense_config['name']}..."
        )
        classifier_for_defense.fit(
            x_train_all,
            y_train_all_categorical,
            batch_size=batch_size,
            nb_epochs=train_epochs,
            verbose=False,
        )
        defense_fn = load_fn(defense_config)
        defense = defense_fn(
            classifier_for_defense, x_train_all, y_train_all_categorical
        )
        _, is_clean = defense.detect_poison(nb_clusters=2, nb_dims=43, reduce="PCA")
        is_clean = np.array(is_clean)
        logger.info(f"Total clean data points: {np.sum(is_clean)}")

        logger.info("Filtering out detected poisoned samples")
        indices_to_keep = is_clean == 1
        x_train_filter = x_train_all[indices_to_keep]
        y_train_filter = y_train_all_categorical[indices_to_keep]
        if len(x_train_filter):
            logger.info(
                f"Fitting model of {model_config['module']}.{model_config['name']}..."
            )
            classifier.fit(
                x_train_filter,
                y_train_filter,
                batch_size=batch_size,
                nb_epochs=train_epochs,
                verbose=False,
            )
        else:
            logger.warning("All data points filtered by defense. Skipping training")

        logger.info(f"Validating on clean test data")
        test_data = load_dataset(
            config["dataset"],
            epochs=1,
            split_type="test",
            preprocessing_fn=preprocessing_fn,
        )
        validation_metric = metrics.MetricList("categorical_accuracy")
        for x, y in tqdm(test_data, desc="Testing"):
            y_pred = classifier.predict(x)
            validation_metric.append(y, y_pred)
        logger.info(f"Unpoisoned validation accuracy: {validation_metric.mean():.2%}")
        results = {"validation_accuracy": validation_metric.mean()}

        if poison_dataset_flag:
            logger.info(f"Testing on poisoned test data")
            test_data = load_dataset(
                config["dataset"],
                epochs=1,
                split_type="test",
                preprocessing_fn=preprocessing_fn,
            )
            test_metric = metrics.MetricList("categorical_accuracy")
            targeted_test_metric = metrics.MetricList("categorical_accuracy")
            for x_test, y_test in tqdm(test_data, desc="Testing"):
                x_test, _ = poison_batch(
                    x_test, y_test, src_class, tgt_class, len(y_test), attack
                )
                y_pred = classifier.predict(x_test)
                test_metric.append(y_test, y_pred)

                y_pred_targeted = y_pred[y_test == src_class]
                if not len(y_pred_targeted):
                    continue
                targeted_test_metric.append(
                    [tgt_class] * len(y_pred_targeted), y_pred_targeted
                )
            results["test_accuracy"] = test_metric.mean()
            results["targeted_misclassification_accuracy"] = targeted_test_metric.mean()
            logger.info(f"Test accuracy: {test_metric.mean():.2%}")
            logger.info(
                f"Test targeted misclassification accuracy: {targeted_test_metric.mean():.2%}"
            )
        return results
