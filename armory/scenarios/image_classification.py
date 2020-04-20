"""
General image classification scenario
"""

import logging

from tqdm import tqdm

from armory.utils.config_loading import (
    load_dataset,
    load_model,
    load_attack,
    load_defense_wrapper,
    load_defense_internal,
)
from armory.utils import metrics
from armory.scenarios.base import Scenario

logger = logging.getLogger(__name__)


class ImageClassificationTask(Scenario):
    def _evaluate(self, config: dict) -> dict:
        """
        Evaluate the config and return a results dict
        """

        model_config = config["model"]
        classifier, preprocessing_fn = load_model(model_config)

        defense_config = config.get("defense") or {}
        defense_type = defense_config.get("type")

        if defense_type in ["Preprocessor", "Postprocessor"]:
            logger.info(f"Applying internal {defense_type} defense to classifier")
            classifier = load_defense_internal(config["defense"], classifier)

        if model_config["fit"]:
            classifier.set_learning_phase(True)
            logger.info(
                f"Fitting model {model_config['module']}.{model_config['name']}..."
            )
            fit_kwargs = model_config["fit_kwargs"]

            logger.info(f"Loading train dataset {config['dataset']['name']}...")
            train_data = load_dataset(
                config["dataset"],
                epochs=fit_kwargs["nb_epochs"],
                split_type="train",
                preprocessing_fn=preprocessing_fn,
            )
            if defense_type == "Trainer":
                logger.info(f"Training with {defense_type} defense...")
                defense = load_defense_wrapper(config["defense"], classifier)
                defense.fit_generator(train_data, **fit_kwargs)
            else:
                logger.info(f"Fitting classifier on clean train dataset...")
                classifier.fit_generator(train_data, **fit_kwargs)

        if defense_type == "Transform":
            # NOTE: Transform currently not supported
            logger.info(f"Transforming classifier with {defense_type} defense...")
            defense = load_defense_wrapper(config["defense"], classifier)
            classifier = defense()

        classifier.set_learning_phase(False)

        # Evaluate the ART classifier on benign test examples
        logger.info(f"Loading test dataset {config['dataset']['name']}...")
        test_data_generator = load_dataset(
            config["dataset"],
            epochs=1,
            split_type="test",
            preprocessing_fn=preprocessing_fn,
        )
        logger.info("Running inference on benign examples...")
        metrics_logger = metrics.MetricsLogger.from_config(config["metric"])

        for x, y in tqdm(test_data_generator, desc="Benign"):
            y_pred = classifier.predict(x)
            metrics_logger.update_task(y, y_pred)
        metrics_logger.log_task()

        # Evaluate the ART classifier on adversarial test examples
        logger.info("Generating / testing adversarial examples...")

        attack = load_attack(config["attack"], classifier)
        test_data_generator = load_dataset(
            config["dataset"],
            epochs=1,
            split_type="test",
            preprocessing_fn=preprocessing_fn,
        )
        for x, y in tqdm(test_data_generator, desc="Attack"):
            x_adv = attack.generate(x=x)
            y_pred_adv = classifier.predict(x_adv)
            metrics_logger.update_task(y, y_pred_adv, adversarial=True)
            metrics_logger.update_perturbation(x, x_adv)
        metrics_logger.log_task(adversarial=True)
        return metrics_logger.results()
