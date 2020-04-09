"""
Classifier evaluation within ARMORY
"""

import json
import os
import sys
import logging
import coloredlogs
from importlib import import_module

import numpy as np

from armory.utils.config_loading import load_dataset, load_model
from armory import paths

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
coloredlogs.install(logging.INFO)


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


def evaluate_classifier(config_path: str) -> None:
    """
    Evaluate a config file for classiifcation robustness against attack.
    """
    with open(config_path) as f:
        config = json.load(f)

    model_config = config["model"]
    classifier, preprocessing_fn = load_model(model_config)

    logger.info(f"Loading dataset {config['dataset']['name']}...")
    train_epochs = config["adhoc"]["train_epochs"]
    src = config["adhoc"]["source_class"]
    tgt = config["adhoc"]["target_class"]

    # Train on training data - could be clean or poisoned
    # and validation on clean data
    train_data_generator = load_dataset(
        config["dataset"],
        epochs=train_epochs,
        split_type="train",
        preprocessing_fn=preprocessing_fn,
    )
    # For poisoned dataset, change to validation_data_generator
    # using split_type="val"
    test_data_generator = load_dataset(
        config["dataset"],
        epochs=train_epochs,
        split_type="test",
        preprocessing_fn=preprocessing_fn,
    )

    # Generate poison examples
    # Ignore this section if using existing poisoned dataset

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

    logger.info(f"Fitting model of {model_config['module']}.{model_config['name']}...")
    for e in range(train_epochs):
        logger.info("Epoch: {}".format(e))

        # train
        for _ in range(train_data_generator.batches_per_epoch):
            x_train, y_train = train_data_generator.get_batch()
            if config["adhoc"]["poison_dataset"]:
                x_train, y_train = poison_batch(
                    x_train, y_train, src, tgt, len(y_train), attack
                )

            classifier.fit(
                x_train, y_train, batch_size=len(y_train), nb_epochs=1, verbose=False,
            )

        # validate on clean data
        correct = 0
        cnt = 0
        for _ in range(test_data_generator.batches_per_epoch):
            x_test, y_test = test_data_generator.get_batch()
            y = classifier.predict(x_test)
            correct += np.sum(np.argmax(y, 1) == y_test)
            cnt += len(y_test)
        validation_accuracy = float(correct) / cnt
        logger.info(f"Unpoisoned validation accuracy: {validation_accuracy:.2%}")

    # Evaluate on test examples - clean or poisoned
    test_data_generator = load_dataset(
        config["dataset"],
        epochs=1,
        split_type="test",
        preprocessing_fn=preprocessing_fn,
    )

    correct = 0
    cnt = 0
    for _ in range(test_data_generator.batches_per_epoch):
        x_test, y_test = test_data_generator.get_batch()
        if config["adhoc"]["poison_dataset"]:
            x_test, _ = poison_batch(x_test, y_test, src, tgt, len(y_test), attack)
        y = classifier.predict(x_test)
        correct += np.sum(np.argmax(y, 1) == y_test)
        cnt += len(y_test)
    test_accuracy = float(correct) / cnt
    logger.info(f"Test accuracy: {test_accuracy:.2%}")

    # Saving results
    logger.info("Saving json output...")
    filepath = os.path.join(paths.docker().output_dir, "gtsrb-evaluation-results.json")
    with open(filepath, "w") as f:
        output_dict = {
            "config": config,
            "results": {
                "validation_accuracy": str(validation_accuracy),
                "test_accuracy": str(test_accuracy),
            },
        }
        json.dump(output_dict, f, sort_keys=True, indent=4)
    logger.info(
        f"Evaluation Results written {paths.docker().output_dir}/gtsrb-evaluation-results.json"
    )


if __name__ == "__main__":
    config_path = sys.argv[-1]
    evaluate_classifier(config_path)
