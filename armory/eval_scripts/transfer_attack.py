"""
Classifier evaluation within ARMORY
"""

import json
import os
import sys
import logging

import numpy as np

from armory.paths import DockerPaths
from armory.utils.config_loading import load_dataset, load_model

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def evaluate_classifier(config_path: str) -> None:
    """
    Evaluate a config file for classiifcation robustness against attack.
    """
    docker_paths = DockerPaths()

    with open(config_path, "r") as fp:
        config = json.load(fp)

    model_config = config["model"]
    classifier, preprocessing_fn = load_model(model_config)

    logger.info(f"Loading dataset {config['dataset']['name']}...")
    clean_x, adv_x, labels = load_dataset(
        config["dataset"], preprocessing_fn=preprocessing_fn
    )

    # Evaluate the ART classifier on benign test examples
    logger.info("Predicting on clean dataset...")
    predictions = classifier.predict(clean_x)
    benign_accuracy = np.sum(np.argmax(predictions, axis=1) == labels) / len(labels)
    logger.info("Accuracy on benign test examples: {}%".format(benign_accuracy * 100))

    # Evaluate the ART classifier on adversarial examples from transfer attack
    logger.info("Predicting on adversarial dataset...")
    predictions = classifier.predict(adv_x)
    adversarial_accuracy = np.sum(np.argmax(predictions, axis=1) == labels) / len(
        labels
    )
    logger.info(
        "Accuracy on adversarial test examples: {}%".format(adversarial_accuracy * 100)
    )

    logger.info("Saving json output...")
    filepath = os.path.join(docker_paths.output_dir, "evaluation-results.json")
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
