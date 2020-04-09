"""
Librispeech Dev Clean scenario evaluation
"""

import logging

from tqdm import tqdm

from armory.utils.config_loading import (
    load_dataset,
    load_model,
    load_attack,
)
from armory.utils import metrics
from armory.scenarios.base import Scenario

logger = logging.getLogger(__name__)


class LibrispeechDevClean(Scenario):
    def _evaluate(self, config: dict) -> dict:
        """
        Evaluate the config and return a results dict
        """

        model_config = config["model"]
        classifier, preprocessing_fn = load_model(model_config)

        if not model_config["weights_file"]:
            classifier.set_learning_phase(True)
            logger.info(
                f"Fitting model {model_config['module']}.{model_config['name']}..."
            )
            fit_kwargs = model_config["fit_kwargs"]
            train_data = load_dataset(
                config["dataset"],
                epochs=fit_kwargs["nb_epochs"],
                split_type="train",
                preprocessing_fn=preprocessing_fn,
            )
            classifier.fit_generator(train_data, **fit_kwargs)

        classifier.set_learning_phase(False)
        # Evaluate the ART classifier on benign test examples
        logger.info(f"Loading dataset {config['dataset']['name']}...")
        test_data_generator = load_dataset(
            config["dataset"],
            epochs=1,
            split_type="test",
            preprocessing_fn=preprocessing_fn,
        )
        logger.info("Running inference on benign examples...")

        task_metric = metrics.categorical_accuracy

        benign_accuracies = []
        for x, y in tqdm(test_data_generator, desc="Benign"):
            y_pred = classifier.predict(x)
            benign_accuracies.extend(task_metric(y, y_pred))
        benign_accuracy = sum(benign_accuracies) / test_data_generator.size
        logger.info(f"Accuracy on benign test examples: {benign_accuracy:.2%}")

        # Evaluate the ART classifier on adversarial test examples
        logger.info("Generating / testing adversarial examples...")

        attack = load_attack(config["attack"], classifier)
        test_data_generator = load_dataset(
            config["dataset"],
            epochs=1,
            split_type="test",
            preprocessing_fn=preprocessing_fn,
        )
        adversarial_accuracies = []
        for x, y in tqdm(test_data_generator, desc="Attack"):
            x_adv = attack.generate(x=x)
            y_pred_adv = classifier.predict(x_adv)
            adversarial_accuracies.extend(task_metric(y, y_pred_adv))
        adversarial_accuracy = sum(adversarial_accuracies) / test_data_generator.size
        logger.info(
            f"Accuracy on adversarial test examples: {adversarial_accuracy:.2%}"
        )
        results = {
            "mean_benign_accuracy": benign_accuracy,
            "mean_adversarial_accuracy": adversarial_accuracy,
        }
        return results
