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
    load_adversarial_dataset,
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
            else:
                logger.info("Fitting classifier on clean train dataset...")

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
        test_data = load_dataset(
            config["dataset"],
            epochs=1,
            split_type="test",
            preprocessing_fn=preprocessing_fn,
        )

        logger.info("Running inference on benign examples...")
        metrics_logger = metrics.MetricsLogger.from_config(config["metric"])

        for x_batch, y_batch in tqdm(test_data, desc="Benign"):
            for x, y in zip(x_batch, y_batch):
                # combine predictions across all stacks
                y_pred = np.mean(classifier.predict(x, batch_size=1), axis=0)
                metrics_logger.update_task(y, y_pred)
        metrics_logger.log_task()

        # Evaluate the ART classifier on adversarial test examples
        logger.info("Generating or loading / testing adversarial examples...")

        attack_config = config["attack"]
        attack_type = attack_config.get("type")
        targeted = bool(attack_config.get("kwargs", {}).get("targeted"))
        if targeted and attack_config.get("use_label"):
            raise ValueError("Targeted attacks cannot have 'use_label'")
        if attack_type == "preloaded":
            test_data = load_adversarial_dataset(
                attack_config,
                epochs=1,
                split_type="adversarial",
                preprocessing_fn=preprocessing_fn,
            )
        else:
            attack = load_attack(attack_config, classifier)
            attack.set_params(batch_size=1)
            test_data = load_dataset(
                config["dataset"],
                epochs=1,
                split_type="test",
                preprocessing_fn=preprocessing_fn,
            )
        for x_batch, y_batch in tqdm(test_data, desc="Attack"):
            if attack_type == "preloaded":
                x_batch = list(zip(*x_batch))
                if targeted:
                    y_batch = list(zip(*y_batch))
            for x, y in zip(x_batch, y_batch):
                if attack_type == "preloaded":
                    x, x_adv = x
                    if targeted:
                        y, y_target = y
                else:
                    # each x is of shape (n_stack, 3, 16, 112, 112)
                    #    n_stack varies
                    if attack_config.get("use_label"):
                        x_adv = attack.generate(x=x, y=y)
                    elif targeted:
                        raise NotImplementedError(
                            "Requires generation of target labels"
                        )
                        # x_adv = attack.generate(x=x, y=y_target)
                    else:
                        x_adv = attack.generate(x=x)
                # combine predictions across all stacks
                y_pred_adv = np.mean(classifier.predict(x_adv, batch_size=1), axis=0)
                if targeted:
                    # NOTE: assumes y != y_target
                    metrics_logger.update_task(y_target, y_pred_adv, adversarial=True)
                else:
                    metrics_logger.update_task(y, y_pred_adv, adversarial=True)
                metrics_logger.update_perturbation([x], [x_adv])
        metrics_logger.log_task(adversarial=True, targeted=targeted)
        return metrics_logger.results()
