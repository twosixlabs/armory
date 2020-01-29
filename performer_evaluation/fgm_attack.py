"""
Classifier evaluation within ARMORY
"""

import json
import sys
import logging
from importlib import import_module

import numpy as np

from art.attacks import FastGradientMethod
from armory.data.data import SUPPORTED_DATASETS
from armory.eval.export import Export


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def evaluate_classifier(config_path: str) -> None:
    """
    Evaluate a config file for classiifcation robustness against attack.
    """
    batch_size = 64
    epochs = 3

    with open(config_path, "r") as fp:
        config = json.load(fp)

    classifier_module = import_module(config["model_file"])
    classifier = getattr(classifier_module, config["model_name"])
    preprocessing_fn = getattr(classifier_module, "preprocessing_fn")

    train_x, train_y, test_x, test_y = SUPPORTED_DATASETS[config["data"]](
        preprocessing_fn=preprocessing_fn
    )

    classifier.fit(train_x, train_y, batch_size=batch_size, nb_epochs=epochs)

    # Evaluate the ART classifier on benign test examples
    predictions = classifier.predict(test_x)
    benign_accuracy = np.sum(np.argmax(predictions, axis=1) == test_y) / len(test_y)
    logger.info("Accuracy on benign test examples: {}%".format(benign_accuracy * 100))

    # Generate adversarial test examples
    attack = FastGradientMethod(classifier=classifier, eps=0.2)
    test_x_adv = attack.generate(x=test_x)

    # Evaluate the ART classifier on adversarial test examples
    predictions = classifier.predict(test_x_adv)
    adversarial_accuracy = np.sum(np.argmax(predictions, axis=1) == test_y) / len(
        test_y
    )
    logger.info(
        "Accuracy on adversarial test examples: {}%".format(adversarial_accuracy * 100)
    )

    exporter = Export(
        performer=config["performer_name"],
        baseline_accuracy=str(benign_accuracy),
        adversarial_accuracy=str(adversarial_accuracy),
    )
    exporter.save()
    logger.info("Evaluation Results written to `outputs/evaluation-results.json")


if __name__ == "__main__":
    config_path = sys.argv[-1]
    evaluate_classifier(config_path)
