"""
Classifier evaluation within ARMORY

Scenario Contributor: MITRE Corporation
"""

import logging
from typing import Optional

from tqdm import tqdm

from armory.utils.config_loading import (
    load_dataset,
    load_model,
    load_attack,
    load_adversarial_dataset,
    load_label_targeter,
)
from armory.utils import metrics
from armory.scenarios.base import Scenario
from armory.data.datasets import ucf101_dataset_canonical_preprocessing

logger = logging.getLogger(__name__)


class Ucf101(Scenario):
    def _evaluate(
        self, config: dict, num_eval_batches: Optional[int], skip_benign: Optional[bool]
    ) -> dict:
        """
        Evaluate the config and return a results dict
        """
        if config["model"]["fit"]:
            raise NotImplementedError("Skipping model training for now")
        if config["dataset"]["batch_size"] != 1:
            raise NotImplementedError("Currently working only with batch size = 1")

        classifier, _ = load_model(config["model"])

        if config["model"]["fit"]:
            classifier.set_learning_phase(True)
            logger.info(
                f"Fitting model {config['model']['module']}.{config['model']['name']}..."
            )
            fit_kwargs = config["model"]["fit_kwargs"]

            logger.info(f"Loading train dataset {config['dataset']['name']}...")
            batch_size = config["dataset"].pop("batch_size")
            config["dataset"]["batch_size"] = config.get("adhoc", {}).get(
                "fit_batch_size", batch_size
            )
            train_data = load_dataset(
                config["dataset"],
                epochs=fit_kwargs["nb_epochs"],
                split_type="train",
                shuffle_files=True,
            )
            config["dataset"]["batch_size"] = batch_size

            logger.info("Fitting classifier on clean train dataset...")
            classifier.fit_generator(train_data, **fit_kwargs)
            # TODO: make the training process much more robust!
            # Need to make it easy to do custom training

            # TODO: save model weights

        classifier.set_learning_phase(False)

        metrics_logger = metrics.MetricsLogger.from_config(
            config["metric"], skip_benign=skip_benign
        )
        if skip_benign:
            logger.info("Skipping benign classification...")
        else:
            # Evaluate the ART classifier on benign test examples
            logger.info(f"Loading test dataset {config['dataset']['name']}...")
            test_data = load_dataset(
                config["dataset"],
                epochs=1,
                split_type="test",
                num_batches=num_eval_batches,
                shuffle_files=False,
            )

            logger.info("Running inference on benign examples...")
            for x, y in tqdm(test_data, desc="Benign"):
                # Ensure that input sample isn't overwritten by classifier
                x.flags.writeable = False
                with metrics.resource_context(
                    name="Inference",
                    profiler=config["metric"].get("profiler_type"),
                    computational_resource_dict=metrics_logger.computational_resource_dict,
                ):
                    y_pred = classifier.predict(x)
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
                preprocessing_fn=ucf101_dataset_canonical_preprocessing,
                num_batches=num_eval_batches,
                shuffle_files=False,
            )
        else:
            attack = load_attack(attack_config, classifier)
            if targeted != getattr(attack, "targeted", False):
                logger.warning(
                    f"targeted config {targeted} != attack field {getattr(attack, 'targeted', False)}"
                )
            # attack.set_params(batch_size=1)  # TODO: do we still need this?
            test_data = load_dataset(
                config["dataset"],
                epochs=1,
                split_type="test",
                num_batches=num_eval_batches,
                shuffle_files=False,
            )
            if targeted:
                label_targeter = load_label_targeter(attack_config["targeted_labels"])
        for x, y in tqdm(test_data, desc="Attack"):
            with metrics.resource_context(
                name="Attack",
                profiler=config["metric"].get("profiler_type"),
                computational_resource_dict=metrics_logger.computational_resource_dict,
            ):
                if attack_type == "preloaded":
                    x, x_adv = x
                    if targeted:
                        y, y_target = y
                elif attack_config.get("use_label"):
                    x_adv = attack.generate(x=x, y=y)
                elif targeted:
                    y_target = label_targeter.generate(y)
                    x_adv = attack.generate(x=x, y=y_target)
                else:
                    x_adv = attack.generate(x=x)

            # Ensure that input sample isn't overwritten by classifier
            x_adv.flags.writeable = False
            y_pred_adv = classifier.predict(x_adv)
            if targeted:
                metrics_logger.update_task(y_target, y_pred_adv, adversarial=True)
            else:
                metrics_logger.update_task(y, y_pred_adv, adversarial=True)
            metrics_logger.update_perturbation([x], [x_adv])
        metrics_logger.log_task(adversarial=True, targeted=targeted)
        return metrics_logger.results()
