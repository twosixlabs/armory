"""
Classifier evaluation within ARMORY
"""

import json
import sys
import numpy as np
from armory.webapi.data import SUPPORTED_DATASETS
from art.attacks import FastGradientMethod
from importlib import import_module
from armory.eval.export import Export


import logging
import coloredlogs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
coloredlogs.install()


def evaluate_classifier(config_path):
    """
    Evaluate a config file for classiifcation robustness against attack.
    """

    # TODO: Preprocessing needs to be refactored elsewhere!
    def _normalize_img(img_batch):
        norm_batch = img_batch.astype(np.float32) / 255.0
        return norm_batch

    with open(config_path, "r") as fp:
        config = json.load(fp)

    classifier_module = import_module(config["model_file"])
    classifier = getattr(classifier_module, config["model_name"])

    train_ds, test_ds = SUPPORTED_DATASETS[config["data"]]()

    x_train, y_train = train_ds["image"], train_ds["label"]
    x_test, y_test = test_ds["image"], test_ds["label"]
    x_train, x_test = _normalize_img(x_train), _normalize_img(x_test)

    classifier.fit(
        x_train, y_train, batch_size=64, nb_epochs=3,
    )

    # Evaluate the ART classifier on benign test examples
    predictions = classifier.predict(x_test)
    benign_accuracy = np.sum(np.argmax(predictions, axis=1) == y_test) / len(y_test)
    logger.info("Accuracy on benign test examples: {}%".format(benign_accuracy * 100))

    # Generate adversarial test examples
    attack = FastGradientMethod(classifier=classifier, eps=0.2)
    x_test_adv = attack.generate(x=x_test)

    # Evaluate the ART classifier on adversarial test examples
    predictions = classifier.predict(x_test_adv)
    adversarial_accuracy = np.sum(np.argmax(predictions, axis=1) == y_test) / len(
        y_test
    )
    logger.info(
        "Accuracy on adversarial test examples: {}%".format(adversarial_accuracy * 100)
    )

    exporter = Export(benign_accuracy, adversarial_accuracy)
    exporter.save()


if __name__ == "__main__":
    config_path = sys.argv[0]
    evaluate_classifier(config_path)
