"""
PyTorch example (naive) defense
"""

import importlib
import json
import logging
import sys

import coloredlogs
import numpy as np

from armory.data import data
from armory.art_experimental import defences as defences_ext


logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.INFO)


def _evaluate_classifier(config: dict) -> None:
    """
    Evaluate a config file for classification robustness against attack.
    """
    classifier_module = importlib.import_module(config["model_file"])
    classifier = getattr(classifier_module, config["model_name"])
    preprocessing_fn = getattr(classifier_module, "preprocessing_fn")
    batch_size = 16

    # defense
    defense = config.get("defense")
    if defense:
        defense_module = importlib.import_module(defense["module"])
        DefenseClass = getattr(defense_module, defense["class"])
        transformer = DefenseClass(**defense.get("kwargs", {}))
        if not isinstance(transformer, defences_ext.Transformer):
            raise ValueError(
                f'{defense["module"]}.{"class"} is not an instance of '
                f"{defences_ext.Transformer}"
            )
        defended_classifier = transformer.transform(classifier)

    # retrofitted to work with existing code

    clean_x, adv_x, labels = data.load(
        config["data"], preprocessing_fn=preprocessing_fn
    )

    logger.debug(f"Original model:\n{classifier}")
    logger.info("Predicting on clean dataset")
    clean_y_pred = classifier.predict(clean_x, batch_size=batch_size)
    clean_accuracy = np.sum(np.argmax(clean_y_pred, axis=1) == labels) / len(labels)
    logger.info(f"Accuracy on benign test examples: {clean_accuracy * 100}%")

    # Evaluate the ART classifier on adversarial examples from transfer attack
    logger.info("Predicting on adversarial dataset...")
    adv_y_pred = classifier.predict(adv_x, batch_size=batch_size)
    adv_accuracy = np.sum(np.argmax(adv_y_pred, axis=1) == labels) / len(labels)
    logger.info(f"Accuracy on adversarial test examples: {adv_accuracy * 100}%")

    # re-evaluate on defended classifier
    if defense:
        logger.debug(f"Defended classifier:\n{defended_classifier}")
        logger.info(f'Classifier defended by {defense["module"]}.{"class"} transform')
        logger.info("Predicting on clean dataset")
        def_clean_y_pred = defended_classifier.predict(clean_x, batch_size=batch_size)
        def_clean_accuracy = np.sum(
            np.argmax(def_clean_y_pred, axis=1) == labels
        ) / len(labels)
        logger.info(f"Accuracy on benign test examples: {def_clean_accuracy * 100}%")

        # Evaluate the ART classifier on adversarial examples from transfer attack
        logger.info("Predicting on adversarial dataset...")
        def_adv_y_pred = defended_classifier.predict(adv_x, batch_size=batch_size)
        def_adv_accuracy = np.sum(np.argmax(def_adv_y_pred, axis=1) == labels) / len(
            labels
        )
        logger.info(f"Accuracy on adversarial test examples: {def_adv_accuracy * 100}%")


def evaluate_classifier(config_path: str) -> None:
    """
    Evaluate a config file for classification robustness against attack.

    Export values.
    """
    with open(config_path) as f:
        config = json.load(f)

    _evaluate_classifier(config)


if __name__ == "__main__":
    config_path = sys.argv[-1]
    evaluate_classifier(config_path)
