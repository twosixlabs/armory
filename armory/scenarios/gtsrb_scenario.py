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

    """
    Train on training data - could be clean or poisoned
    and validation on clean data
    """
    train_data_generator = load_dataset(
        config["dataset"],
        epochs=train_epochs,
        split_type="train",
        preprocessing_fn=preprocessing_fn,
    )
    """
    For poisoned dataset, change to validation_data_generator
    using split_type="val"
    """
    test_data_generator = load_dataset(
        config["dataset"],
        epochs=train_epochs,
        split_type="test",
        preprocessing_fn=preprocessing_fn,
    )

    logger.info(f"Fitting model of {model_config['module']}.{model_config['name']}...")
    for e in range(train_epochs):
        logger.info("Epoch: {}".format(e))

        # train
        classifier.fit_generator(
            train_data_generator, nb_epochs=1, verbose=False,
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
        logger.info("Validation accuracy: {}".format(validation_accuracy))

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
        y = classifier.predict(x_test)
        correct += np.sum(np.argmax(y, 1) == y_test)
        cnt += len(y_test)
    test_accuracy = float(correct) / cnt
    logger.info("Test accuracy: {}".format(test_accuracy))

    """
    Generate poison examples
    Ignore this section if using existing poisoned dataset
    """
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

    logger.info("Generating poisoning examples...")
    test_data_generator = load_dataset(
        config["dataset"],
        epochs=1,
        split_type="test",
        preprocessing_fn=preprocessing_fn,
    )

    """
    In this example, all images of "src" class have a trigger
    added and re-labeled as "tgt" class
    NOTE: currently art.attacks.PoisonAttackBackdoor only supports
      black-white images.  One way to generate poisoned examples
      is to convert each batch of multi-channel images of shape
      (N,W,H,C) to N separate (C,W,H)-tuple, where C would be
      interpreted by PoisonAttackBackdoor as the batch size,
      and each channel would have a backdoor trigger added
    """
    src = 5
    tgt = 42
    poison_imgs = []
    poison_labels = []
    for _ in range(test_data_generator.batches_per_epoch):
        x_test, y_test = test_data_generator.get_batch()
        src_imgs = x_test[y_test == src]
        if len(src_imgs) > 0:
            for src_img in src_imgs:
                src_img = np.transpose(src_img, (2, 0, 1))
                p_img, p_label = attack.poison(src_img, tgt * src_img.shape[0])
                poison_imgs.append(np.transpose(p_img, (1, 2, 0)))
                poison_labels.append(p_label)
    poison_imgs = np.array(poison_imgs)
    poison_labels = np.array(poison_labels)

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
