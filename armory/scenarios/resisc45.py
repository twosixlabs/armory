"""
Classifier evaluation within ARMORY
"""

import logging
from importlib import import_module

import numpy as np

from armory.scenarios.base import Scenario
from armory.utils.config_loading import load_dataset, load_model

logger = logging.getLogger(__name__)


class Resisc45(Scenario):
    def _evaluate(self, config: dict) -> dict:
        """
        Evaluate the config and return a results dict
        """

        model_config = config["model"]
        classifier, preprocessing_fn = load_model(model_config)

        logger.info(f"Loading dataset {config['dataset']['name']}...")
        test_data_generator = load_dataset(
            config["dataset"],
            epochs=1,
            split_type="test",
            preprocessing_fn=preprocessing_fn,
        )

        if not model_config["weights_file"]:
            logger.info(
                f"Fitting clean unpoisoned model of {model_config['module']}.{model_config['name']}..."
            )
            # TODO train here

        # Evaluate the ART classifier on benign test examples
        logger.info("Running inference on benign examples...")
        benign_accuracy = 0
        cnt = 0
        for _ in range(test_data_generator.batches_per_epoch):
            x, y = test_data_generator.get_batch()
            predictions = classifier.predict(x)
            benign_accuracy += np.sum(np.argmax(predictions, axis=1) == y) / len(y)
            cnt += 1
        benign_accuracy = benign_accuracy / cnt
        logger.info(
            "Accuracy on benign test examples: {}%".format(benign_accuracy * 100)
        )

        # Generate adversarial test examples
        attack_config = config["attack"]
        attack_module = import_module(attack_config["module"])
        attack_fn = getattr(attack_module, attack_config["name"])

        test_data_generator = load_dataset(
            config["dataset"],
            epochs=1,
            split_type="test",
            preprocessing_fn=preprocessing_fn,
        )
        # Evaluate the ART classifier on adversarial test examples
        logger.info("Generating / testing adversarial examples...")

        attack = attack_fn(classifier=classifier, **attack_config["kwargs"])
        adversarial_accuracy = 0
        cnt = 0
        for _ in range(test_data_generator.batches_per_epoch):
            x, y = test_data_generator.get_batch()
            test_x_adv = attack.generate(x=x)
            predictions = classifier.predict(test_x_adv)
            adversarial_accuracy += np.sum(np.argmax(predictions, axis=1) == y) / len(y)
            cnt += 1
        adversarial_accuracy = adversarial_accuracy / cnt
        logger.info(
            "Accuracy on adversarial test examples: {}%".format(
                adversarial_accuracy * 100
            )
        )
        results = {
            "baseline_accuracy": str(benign_accuracy),
            "adversarial_accuracy": str(adversarial_accuracy),
        }
        return results
