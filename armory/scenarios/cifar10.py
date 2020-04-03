"""
CIFAR10 scenario evaluation
"""

import logging

import numpy as np

from armory.utils.config_loading import load_dataset, load_model, load_attack
from armory.scenarios import Scenario


logger = logging.getLogger(__name__)


class Cifar10(Scenario):
    def _evaluate(self, config: dict) -> dict:
        """
        Evaluate the config and return a results dict
        """

        model_config = config["model"]
        classifier, preprocessing_fn = load_model(model_config)

        if not model_config["weights_file"]:
            logger.info(
                f"Fitting clean unpoisoned model of {model_config['module']}.{model_config['name']}..."
            )
            fit_kwargs = model_config["fit_kwargs"]
            if "epochs" in fit_kwargs:
                pass
                if "nb_epochs" in fit_kwargs:
                    pass
                else:
                    pass
            else:
                if "nb_epochs" in fit_kwargs:
                    pass
                else:
                    epochs = 2
            train_data = load_dataset(
                config["dataset"],
                epochs=epochs,
                split_type="train",
                preprocessing_fn=preprocessing_fn,
            )
            classifier.fit_generator(train_data, **model_config["fit_kwargs"])
            # TODO train here

        # Evaluate the ART classifier on benign test examples
        logger.info(f"Loading dataset {config['dataset']['name']}...")
        test_data_generator = load_dataset(
            config["dataset"],
            epochs=1,
            split_type="test",
            preprocessing_fn=preprocessing_fn,
        )
        logger.info("Running inference on benign examples...")
        benign_accuracy = 0
        for cnt, (x, y) in test_data_generator:
            predictions = classifier.predict(x)
            benign_accuracy += np.sum(np.argmax(predictions, axis=1) == y)
        benign_accuracy /= test_data_generator.size
        logger.info(
            "Accuracy on benign test examples: {}%".format(benign_accuracy * 100)
        )

        # Evaluate the ART classifier on adversarial test examples
        logger.info("Generating / testing adversarial examples...")

        attack = load_attack(config["attack"], classifier)
        test_data_generator = load_dataset(
            config["dataset"],
            epochs=1,
            split_type="test",
            preprocessing_fn=preprocessing_fn,
        )
        adversarial_accuracy = 0
        for cnt, (x, y) in test_data_generator:
            test_x_adv = attack.generate(x=x)
            predictions = classifier.predict(test_x_adv)
            adversarial_accuracy += np.sum(np.argmax(predictions, axis=1) == y)
        adversarial_accuracy /= test_data_generator.size
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
