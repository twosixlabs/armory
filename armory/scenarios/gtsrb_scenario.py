"""
Classifier evaluation within ARMORY

Scenario Contributor: MITRE Corporation
"""

import logging
from importlib import import_module

import numpy as np
from tqdm import tqdm
from tensorflow.keras.utils import to_categorical

from armory.scenarios.base import Scenario
from armory.utils.config_loading import load_dataset, load_model, load_defense_internal

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

        defense = config.get("defense", {})
        defense_type = defense.get("type")

        if defense_type in ["Preprocessor", "Postprocessor"]:
            logger.info(f"Applying internal {defense_type} defense to classifier")
            classifier = load_defense_internal(defense, classifier)

        logger.info(f"Loading dataset {config['dataset']['name']}...")
        train_epochs = config["adhoc"]["train_epochs"]
        src_class = config["adhoc"]["source_class"]
        tgt_class = config["adhoc"]["target_class"]
        fraction_poisoned = config["adhoc"]["fraction_poisoned"]
        poison_dataset_flag = config["adhoc"]["poison_dataset"]
        batch_size = config["dataset"]["batch_size"]

        train_data_generator = load_dataset(
            config["dataset"],
            epochs=1,
            split_type="train",
            preprocessing_fn=preprocessing_fn,
        )

        attack_config = config["attack"]
        attack_module = import_module(attack_config["module"])
        attack_fn = getattr(attack_module, attack_config["name"])
        poison_module = import_module(attack_config["kwargs"]["poison_module"])
        poison_type = attack_config["kwargs"]["poison_type"]

        def add_modification(x):
            if poison_type == "pattern":
                poison_fn = getattr(poison_module, "add_pattern_bd")
                return poison_fn(x, pixel_value=255)
            elif poison_type == "pixel":
                poison_fn = getattr(poison_module, "add_single_bd")
                return poison_fn(x, pixel_value=255)
            elif poison_type == "image":
                poison_fn = getattr(poison_module, "insert_image")
                return poison_fn(x, backdoor_path="PATH_TO_IMG", size=(10, 10))
            else:
                raise ValueError("Unknown backdoor type")

        attack = attack_fn(add_modification)

        x_train_all, y_train_all = [], []
        for x_train, y_train in train_data_generator:

            if poison_dataset_flag:
                poison_batch_flag = np.random.rand() < fraction_poisoned
                if poison_batch_flag:
                    x_train, y_train = poison_batch(
                        x_train, y_train, src_class, tgt_class, len(y_train), attack
                    )
            x_train_all.append(x_train)
            y_train_all.append(y_train)

        x_train_all = np.concatenate(x_train_all, axis=0)
        y_train_all = np.concatenate(y_train_all, axis=0)
        y_train_all_categorical = to_categorical(y_train_all)

        defense_config = config["defense"]
        defense_module = import_module(defense_config["module"])
        defense_fn = getattr(defense_module, defense_config["name"])

        defense = defense_fn(classifier, x_train_all, y_train_all_categorical)
        _, is_clean_lst = defense.detect_poison(nb_clusters=2, nb_dims=43, reduce="PCA")

        is_clean = np.array(is_clean_lst)
        logger.info(f"Total clean data points: {np.sum(is_clean)}")

        indices_to_keep = is_clean == 1
        x_train_filter = x_train_all[indices_to_keep]
        y_train_filter = y_train_all[indices_to_keep]

        classifier.fit(
            x_train_filter,
            y_train_filter,
            batch_size=batch_size,
            nb_epochs=train_epochs,
            verbose=False,
        )

        logger.info(
            f"Fitting model of {model_config['module']}.{model_config['name']}..."
        )
        # Clean test data to be poisoned
        test_data_generator = load_dataset(
            config["dataset"],
            epochs=1,
            split_type="test",
            preprocessing_fn=preprocessing_fn,
        )
        # Validate on clean test data
        correct = 0
        cnt = 0
        for x_test, y_test in tqdm(test_data_generator, desc="Testing"):
            y = classifier.predict(x_test)
            correct += np.sum(np.argmax(y, 1) == y_test)
            cnt += len(y_test)
        validation_accuracy = float(correct) / cnt
        logger.info(f"Unpoisoned validation accuracy: {validation_accuracy:.2%}")

        test_data_generator = load_dataset(
            config["dataset"],
            epochs=1,
            split_type="test",
            preprocessing_fn=preprocessing_fn,
        )
        correct = 0
        cnt = 0
        for x_test, y_test in tqdm(test_data_generator, desc="Testing"):
            if poison_dataset_flag:
                x_test, _ = poison_batch(
                    x_test, y_test, src_class, tgt_class, len(y_test), attack
                )
            y = classifier.predict(x_test)
            correct += np.sum(np.argmax(y, 1) == y_test)
            cnt += len(y_test)
        test_accuracy = float(correct) / cnt
        logger.info(f"Test accuracy: {test_accuracy:.2%}")

        results = {
            "validation_accuracy": str(validation_accuracy),
            "test_accuracy": str(test_accuracy),
        }
        if poison_dataset_flag:
            test_data_generator = load_dataset(
                config["dataset"],
                epochs=1,
                split_type="test",
                preprocessing_fn=preprocessing_fn,
            )
            correct = 0
            cnt = 0
            for x_test, y_test in tqdm(test_data_generator, desc="Testing"):
                x_test, _ = poison_batch(
                    x_test, y_test, src_class, tgt_class, len(y_test), attack
                )
                x_test_targeted = x_test[y_test == src_class]
                if len(x_test_targeted) == 0:
                    continue
                y = classifier.predict(x_test_targeted)
                correct += np.sum(np.argmax(y, 1) == tgt_class)
                cnt += len(y)
            targeted_accuracy = float(correct) / cnt
            logger.info(
                f"Test targeted misclassification accuracy: {targeted_accuracy:.2%}"
            )
            results["targeted_misclassification_accuracy"] = str(targeted_accuracy)
        return results
