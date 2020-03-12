"""
Classifier evaluation within ARMORY
"""

import json
import os
import sys
import logging
from importlib import import_module

import numpy as np

from armory.utils.config_loading import load_dataset, load_model
from armory.paths import DockerPaths

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
    train_epochs = config["adhoc"]["train_epochs"]
    train_data_generator = load_dataset(
        config["dataset"],
        epochs=train_epochs,
        split_type="train",
        preprocessing_fn=preprocessing_fn,
    )
    test_data_generator = load_dataset(
        config["dataset"],
        epochs=2,
        split_type="test",
        preprocessing_fn=preprocessing_fn,
    )

    logger.info(
        f"Fitting clean unpoisoned model of {model_config['module']}.{model_config['name']}..."
    )

    classifier.fit_generator(
        train_data_generator, nb_epochs=train_data_generator.total_iterations,
    )

    # Evaluate the ART classifier on benign test examples
    logger.info("Running inference on benign examples...")
    benign_accuracy = 0
    cnt = 0
    for _ in range(test_data_generator.total_iterations // 2):
        x, y = test_data_generator.get_batch()
        predictions = classifier.predict(x)
        benign_accuracy += np.sum(np.argmax(predictions, axis=1) == y) / len(y)
        cnt += 1
    logger.info(
        "Accuracy on benign test examples: {}%".format(benign_accuracy * 100 / cnt)
    )

    # Generate adversarial test examples
    attack_config = config["attack"]
    attack_module = import_module(attack_config["module"])
    attack_fn = getattr(attack_module, attack_config["name"])

    # Evaluate the ART classifier on adversarial test examples
    logger.info("Generating / testing adversarial examples...")

    attack = attack_fn(classifier=classifier, **attack_config["kwargs"])
    adversarial_accuracy = 0
    cnt = 0
    for _ in range(test_data_generator.total_iterations // 2):
        x, y = test_data_generator.get_batch()
        test_x_adv = attack.generate(x=x)
        predictions = classifier.predict(test_x_adv)
        adversarial_accuracy += np.sum(np.argmax(predictions, axis=1) == y) / len(y)
        cnt += 1
    logger.info(
        "Accuracy on adversarial test examples: {}%".format(
            adversarial_accuracy * 100 / cnt
        )
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
