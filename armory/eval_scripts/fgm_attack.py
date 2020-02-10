"""
Classifier evaluation within ARMORY
"""

import json
import os
import sys
import logging
from importlib import import_module

import numpy as np

from armory.data import datasets
from armory import paths

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def evaluate_classifier(config_path: str) -> None:
    """
    Evaluate a config file for classiifcation robustness against attack.
    """
    with open(config_path, "r") as fp:
        config = json.load(fp)

    model_config = config["model"]
    classifier_module = import_module(model_config["module"])
    classifier_fn = getattr(classifier_module, model_config["name"])
    classifier = classifier_fn(
        model_config["model_kwargs"], model_config["wrapper_kwargs"]
    )

    preprocessing_fn = getattr(classifier_module, "preprocessing_fn")

    logger.info(f"Loading dataset {config['dataset']['name']}...")
    train_x, train_y, test_x, test_y = datasets.load(
        config["dataset"]["name"], preprocessing_fn=preprocessing_fn
    )

    logger.info(
        f"Fitting clean unpoisoned model of {model_config['module']}.{model_config['name']}..."
    )
    classifier.fit(
        train_x,
        train_y,
        batch_size=config["adhoc"]["batch_size"],
        nb_epochs=config["adhoc"]["epochs"],
    )

    # Evaluate the ART classifier on benign test examples
    logger.info("Running inference on benign examples...")
    predictions = classifier.predict(test_x)
    benign_accuracy = np.sum(np.argmax(predictions, axis=1) == test_y) / len(test_y)
    logger.info("Accuracy on benign test examples: {}%".format(benign_accuracy * 100))

    # Generate adversarial test examples
    attack_config = config["attack"]
    attack_module = import_module(attack_config["module"])
    attack_fn = getattr(attack_module, attack_config["name"])

    logger.info("Generating adversarial examples...")
    attack = attack_fn(classifier=classifier, **attack_config["kwargs"])
    test_x_adv = attack.generate(x=test_x)

    # Evaluate the ART classifier on adversarial test examples
    logger.info("Running inference on adversarial examples...")
    predictions = classifier.predict(test_x_adv)
    adversarial_accuracy = np.sum(np.argmax(predictions, axis=1) == test_y) / len(
        test_y
    )
    logger.info(
        "Accuracy on adversarial test examples: {}%".format(adversarial_accuracy * 100)
    )

    logger.info("Saving json output...")
    filepath = os.path.join(paths.OUTPUTS, "evaluation-results.json")
    with open(filepath, "w") as f:
        output_dict = {
            "config": config,
            "results": {
                "baseline_accuracy": str(benign_accuracy),
                "adversarial_accuracy": str(adversarial_accuracy),
            },
        }
        json.dump(output_dict, f, sort_keys=True, indent=4)
    logger.info(f"Evaluation Results written to {filepath}")


if __name__ == "__main__":
    config_path = sys.argv[-1]
    evaluate_classifier(config_path)
