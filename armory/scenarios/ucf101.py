"""
Classifier evaluation within ARMORY

Scenario Contributor: MITRE Corporation
"""

import logging

import numpy as np
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


class Ucf101(Scenario):
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
            train_epochs = config["model"]["fit_kwargs"]["nb_epochs"]
            batch_size = config["dataset"]["batch_size"]

            logger.info(f"Loading train dataset {config['dataset']['name']}...")
            train_data = load_dataset(
                config["dataset"],
                epochs=train_epochs,
                split_type="train",
                preprocessing_fn=preprocessing_fn,
            )

            if defense_type == "Trainer":
                logger.info(f"Training with {defense_type} defense...")
                defense = load_defense_wrapper(config["defense"], classifier)

            for epoch in range(train_epochs):
                classifier.set_learning_phase(True)

                for _ in tqdm(
                    range(train_data.batches_per_epoch),
                    desc=f"Epoch: {epoch}/{train_epochs}",
                ):
                    x, y = train_data.get_batch()
                    # x_trains consists of one or more videos, each represented as an
                    # ndarray of shape (n_stacks, 3, 16, 112, 112).
                    # To train, randomly sample a batch of stacks
                    x = np.stack([x_i[np.random.randint(x_i.shape[0])] for x_i in x])
                    if defense_type == "Trainer":
                        defense.fit(x, y, batch_size=batch_size, nb_epochs=1)
                    else:
                        classifier.fit(x, y, batch_size=batch_size, nb_epochs=1)

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

        for x_batch, y_batch in tqdm(test_data_generator, desc="Benign"):
            for x, y in zip(x_batch, y_batch):
                # combine predictions across all stacks
                y_pred = np.mean(classifier.predict(x), axis=0)
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
        for x_batch, y_batch in tqdm(test_data_generator, desc="Attack"):
            for x, y in zip(x_batch, y_batch):
                # each x is of shape (n_stack, 3, 16, 112, 112)
                #    n_stack varies
                attack.set_params(batch_size=x.shape[0])
                x_adv = attack.generate(x=x)
                # combine predictions across all stacks
                y_pred = np.mean(classifier.predict(x), axis=0)
                metrics_logger.update_task(y, y_pred, adversarial=True)
                metrics_logger.update_perturbation([x], [x_adv])
        metrics_logger.log_task(adversarial=True)
        return metrics_logger.results()
