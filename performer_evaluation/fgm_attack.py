"""
Classifier evaluation within ARMORY
"""

import json
import sys

from armory.webapi.data import SUPPORTED_DATASETS
from art.attacks import FastGradientMethod
from importlib import import_module
from armory.eval.export import Export


import logging

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

    train_ds, test_ds, num_train, num_test = SUPPORTED_DATASETS[config["data"]](
        batch_size=batch_size, epochs=epochs, normalize=True
    )
    x_test, y_test = test_ds

    classifier_module = import_module(config["model_file"])
    classifier = getattr(classifier_module, config["model_name"])

    steps_per_epoch = int(num_train / batch_size)
    classifier._model.fit_generator(
        train_ds, epochs=epochs, steps_per_epoch=steps_per_epoch
    )

    # Evaluate the ART classifier on benign test examples
    benign_accuracy = classifier._model.evaluate(x_test, y_test, batch_size=batch_size)
    logger.info(
        "Accuracy on benign test examples: {}%".format(benign_accuracy[1] * 100)
    )

    # Generate adversarial test examples
    attack = FastGradientMethod(classifier=classifier, eps=0.2)
    x_test_adv = attack.generate(x=x_test)

    # Evaluate the ART classifier on adversarial test examples
    adversarial_accuracy = classifier._model.evaluate(x_test_adv, y_test, batch_size=64)
    logger.info(
        "Accuracy on adversarial test examples: {}%".format(
            adversarial_accuracy[1] * 100
        )
    )

    exporter = Export(
        performer=config["performer_name"],
        baseline_accuracy=str(benign_accuracy[1]),
        adversarial_accuracy=str(adversarial_accuracy[1]),
    )
    exporter.save()
    logger.info("Evaluation Results written to `outputs/evaluation-results.json")


if __name__ == "__main__":
    config_path = sys.argv[-1]
    evaluate_classifier(config_path)
