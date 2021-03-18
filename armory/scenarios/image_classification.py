"""
General image recognition scenario for image classification and object detection.
"""

import logging
from typing import Optional
from copy import deepcopy

from tqdm import tqdm
import numpy as np

from armory.utils.config_loading import (
    load_dataset,
    load_model,
    load_attack,
    load_adversarial_dataset,
    load_defense_wrapper,
    load_defense_internal,
    load_label_targeter,
)
from armory.utils import metrics
from armory.scenarios.base import Scenario
from armory.utils.export import SampleExporter

logger = logging.getLogger(__name__)


class ImageClassificationTask(Scenario):
    def _evaluate(
        self,
        config: dict,
        num_eval_batches: Optional[int],
        skip_benign: Optional[bool],
        skip_attack: Optional[bool],
        skip_misclassified: Optional[bool],
    ) -> dict:
        """
        Evaluate the config and return a results dict
        """
        model_config = config["model"]
        estimator, _ = load_model(model_config)

        defense_config = config.get("defense") or {}
        defense_type = defense_config.get("type")

        if defense_type in ["Preprocessor", "Postprocessor"]:
            logger.info(f"Applying internal {defense_type} defense to estimator")
            estimator = load_defense_internal(config["defense"], estimator)

        if model_config["fit"]:
            try:
                logger.info(
                    f"Fitting model {model_config['module']}.{model_config['name']}..."
                )
                fit_kwargs = model_config["fit_kwargs"]

                logger.info(f"Loading train dataset {config['dataset']['name']}...")
                train_data = load_dataset(
                    config["dataset"],
                    epochs=fit_kwargs["nb_epochs"],
                    split=config["dataset"].get("train_split", "train"),
                    shuffle_files=True,
                )
                if defense_type == "Trainer":
                    logger.info(f"Training with {defense_type} defense...")
                    defense = load_defense_wrapper(config["defense"], estimator)
                    defense.fit_generator(train_data, **fit_kwargs)
                else:
                    logger.info("Fitting estimator on clean train dataset...")
                    estimator.fit_generator(train_data, **fit_kwargs)
            except NotImplementedError:
                raise NotImplementedError(
                    "Training has not yet been implemented for object detectors"
                )

        if defense_type == "Transform":
            # NOTE: Transform currently not supported
            logger.info(f"Transforming estimator with {defense_type} defense...")
            defense = load_defense_wrapper(config["defense"], estimator)
            estimator = defense()

        attack_config = config["attack"]
        attack_type = attack_config.get("type")

        targeted = bool(attack_config.get("kwargs", {}).get("targeted"))
        metrics_logger = metrics.MetricsLogger.from_config(
            config["metric"],
            skip_benign=skip_benign,
            skip_attack=skip_attack,
            targeted=targeted,
        )

        eval_split = config["dataset"].get("eval_split", "test")
        if skip_benign:
            logger.info("Skipping benign classification...")
        else:
            # Evaluate the ART estimator on benign test examples
            logger.info(f"Loading test dataset {config['dataset']['name']}...")
            test_data = load_dataset(
                config["dataset"],
                epochs=1,
                split=eval_split,
                num_batches=num_eval_batches,
                shuffle_files=False,
            )

            logger.info("Running inference on benign examples...")
            for x, y in tqdm(test_data, desc="Benign"):
                # Ensure that input sample isn't overwritten by estimator
                x.flags.writeable = False
                with metrics.resource_context(
                    name="Inference",
                    profiler=config["metric"].get("profiler_type"),
                    computational_resource_dict=metrics_logger.computational_resource_dict,
                ):
                    y_pred = estimator.predict(x)
                metrics_logger.update_task(y, y_pred)
            metrics_logger.log_task()

        if skip_attack:
            logger.info("Skipping attack generation...")
            return metrics_logger.results()

        # Evaluate the ART estimator on adversarial test examples
        logger.info("Generating or loading / testing adversarial examples...")

        if skip_misclassified:
            acc_task_idx = [i.name for i in metrics_logger.tasks].index(
                "categorical_accuracy"
            )
            benign_acc = metrics_logger.tasks[acc_task_idx].values()

        if targeted and attack_config.get("use_label"):
            raise ValueError("Targeted attacks cannot have 'use_label'")
        if attack_type == "preloaded":
            preloaded_split = attack_config.get("kwargs", {}).get(
                "split", "adversarial"
            )
            test_data = load_adversarial_dataset(
                attack_config,
                epochs=1,
                split=preloaded_split,
                num_batches=num_eval_batches,
                shuffle_files=False,
            )
        else:
            attack = load_attack(attack_config, estimator)
            if targeted != getattr(attack, "targeted", False):
                logger.warning(
                    f"targeted config {targeted} != attack field {getattr(attack, 'targeted', False)}"
                )
            test_data = load_dataset(
                config["dataset"],
                epochs=1,
                split=eval_split,
                num_batches=num_eval_batches,
                shuffle_files=False,
            )
            if targeted:
                label_targeter = load_label_targeter(attack_config["targeted_labels"])

        export_samples = config["scenario"].get("export_samples")
        if export_samples is not None and export_samples > 0:
            sample_exporter = SampleExporter(
                self.scenario_output_dir, test_data.context, export_samples
            )
        else:
            sample_exporter = None

        for batch_idx, (x, y) in enumerate(tqdm(test_data, desc="Attack")):
            with metrics.resource_context(
                name="Attack",
                profiler=config["metric"].get("profiler_type"),
                computational_resource_dict=metrics_logger.computational_resource_dict,
            ):
                if attack_type == "preloaded":
                    if len(x) == 2:
                        x, x_adv = x
                    else:
                        x_adv = x
                    if targeted:
                        y, y_target = y
                else:
                    generate_kwargs = deepcopy(attack_config.get("generate_kwargs", {}))
                    # Temporary workaround for ART code requirement of ndarray mask
                    if "mask" in generate_kwargs:
                        generate_kwargs["mask"] = np.array(generate_kwargs["mask"])
                    if attack_config.get("use_label"):
                        generate_kwargs["y"] = y
                    elif targeted:
                        y_target = label_targeter.generate(y)
                        generate_kwargs["y"] = y_target

                    if skip_misclassified and benign_acc[batch_idx] == 0:
                        x_adv = x
                    else:
                        x_adv = attack.generate(x=x, **generate_kwargs)

            # Ensure that input sample isn't overwritten by estimator
            x_adv.flags.writeable = False
            y_pred_adv = estimator.predict(x_adv)
            metrics_logger.update_task(y, y_pred_adv, adversarial=True)
            if targeted:
                metrics_logger.update_task(
                    y_target, y_pred_adv, adversarial=True, targeted=True
                )
            metrics_logger.update_perturbation(x, x_adv)
            if sample_exporter is not None:
                sample_exporter.export(x, x_adv, y, y_pred_adv)
        metrics_logger.log_task(adversarial=True)
        if targeted:
            metrics_logger.log_task(adversarial=True, targeted=True)
        return metrics_logger.results()
