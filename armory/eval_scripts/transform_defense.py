"""
PyTorch example (naive) defense
"""

import json
import logging
import sys
from importlib import import_module

import coloredlogs
import numpy as np

from armory.data import datasets
from armory.art_experimental import defences as defences_ext


logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.INFO)


def evaluate_classifier(config_path: str) -> None:
    """
    Evaluate a config file for classification robustness against attack.
    """
    with open(config_path) as fp:
        config = json.load(fp)

    model_config = config["model"]
    classifier_module = import_module(model_config["module"])
    classifier_fn = getattr(classifier_module, model_config["name"])
    classifier = classifier_fn(
        model_config["model_kwargs"], model_config["wrapper_kwargs"]
    )

    batch_size = config["adhoc"]["batch_size"]

    preprocessing_fn = getattr(classifier_module, "preprocessing_fn")

    # Defense
    defense_config = config["defense"]
    defense_module = import_module(defense_config["module"])
    defense_fn = getattr(defense_module, defense_config["name"])
    transformer = defense_fn(**defense_config.get("kwargs", {}))
    if not isinstance(transformer, defences_ext.Transformer):
        raise ValueError(
            f'{defense_config["module"]}.{defense_config["name"]} is not an instance of '
            f"{defences_ext.Transformer}"
        )
    defended_classifier = transformer.transform(classifier)

    # retrofitted to work with existing code
    logger.info(f"Loading dataset {config['dataset']['name']}...")
    clean_x, adv_x, labels = datasets.load(
        config["dataset"]["name"], preprocessing_fn=preprocessing_fn
    )

    logger.debug(f"Original model:\n{classifier}")
    logger.info("Predicting on clean dataset...")
    clean_y_pred = classifier.predict(clean_x, batch_size=batch_size)
    clean_accuracy = np.sum(np.argmax(clean_y_pred, axis=1) == labels) / len(labels)
    logger.info(f"Accuracy on benign test examples: {clean_accuracy * 100}%")

    # Evaluate the ART classifier on adversarial examples from transfer attack
    logger.info("Predicting on adversarial dataset...")
    adv_y_pred = classifier.predict(adv_x, batch_size=batch_size)
    adv_accuracy = np.sum(np.argmax(adv_y_pred, axis=1) == labels) / len(labels)
    logger.info(f"Accuracy on adversarial test examples: {adv_accuracy * 100}%")

    # Ee-evaluate on defended classifier
    logger.debug(f"Defended classifier:\n{defended_classifier}")
    logger.info(
        f'Classifier defended by {defense_config["module"]}.{defense_config["name"]} transform'
    )
    logger.info("Predicting on clean dataset...")
    def_clean_y_pred = defended_classifier.predict(clean_x, batch_size=batch_size)
    def_clean_accuracy = np.sum(np.argmax(def_clean_y_pred, axis=1) == labels) / len(
        labels
    )
    logger.info(f"Accuracy on benign test examples: {def_clean_accuracy * 100}%")

    # Evaluate the ART classifier on adversarial examples from transfer attack
    logger.info("Predicting on adversarial dataset...")
    def_adv_y_pred = defended_classifier.predict(adv_x, batch_size=batch_size)
    def_adv_accuracy = np.sum(np.argmax(def_adv_y_pred, axis=1) == labels) / len(labels)
    logger.info(f"Accuracy on adversarial test examples: {def_adv_accuracy * 100}%")


if __name__ == "__main__":
    config_path = sys.argv[-1]
    evaluate_classifier(config_path)
