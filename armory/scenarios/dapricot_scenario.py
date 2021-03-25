"""
General image recognition scenario for image classification and object detection.
"""

import logging
from typing import Optional
from copy import deepcopy

from tqdm import tqdm

from armory.utils.config_loading import (
    load_dataset,
    load_model,
    load_attack,
    load_defense_wrapper,
    load_defense_internal,
    load_label_targeter,
)
from armory.utils import metrics
from armory.scenarios.base import Scenario
from armory.utils.export import SampleExporter

logger = logging.getLogger(__name__)


class ObjectDetectionTask(Scenario):
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
        if skip_misclassified:
            raise ValueError(
                "skip_misclassified shouldn't be set for D-APRICOT scenario"
            )
        if skip_attack:
            raise ValueError("--skip-attack should not be set for D-APRICOT scenario.")
        if skip_benign:
            logger.warning(
                "--skip-benign is being ignored since the D-APRICOT"
                " scenario doesn't include benign evaluation."
            )
        attack_config = config["attack"]
        attack_type = attack_config.get("type")
        if attack_type == "preloaded":
            raise ValueError(
                "D-APRICOT scenario should not have preloaded set to True in attack config"
            )
        elif "targeted_labels" not in attack_config:
            raise ValueError(
                "Attack config must have 'targeted_labels' key, as the "
                "D-APRICOT threat model is targeted."
            )
        elif attack_config.get("use_label"):
            raise ValueError(
                "The D-APRICOT scenario threat model is targeted, and"
                " thus 'use_label' should be set to false."
            )

        if config["dataset"].get("batch_size") != 1:
            raise ValueError("batch_size of 1 is required for D-APRICOT scenario")

        model_config = config["model"]
        estimator, _ = load_model(model_config)

        defense_config = config.get("defense") or {}
        defense_type = defense_config.get("type")

        label_targeter = load_label_targeter(attack_config["targeted_labels"])

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

        metrics_logger = metrics.MetricsLogger.from_config(
            config["metric"], skip_benign=True, skip_attack=False, targeted=True,
        )

        eval_split = config["dataset"].get("eval_split", "test")

        # Evaluate the ART estimator on adversarial test examples
        logger.info("Generating or loading / testing adversarial examples...")

        attack = load_attack(attack_config, estimator)
        test_data = load_dataset(
            config["dataset"],
            epochs=1,
            split=eval_split,
            num_batches=num_eval_batches,
            shuffle_files=False,
        )

        export_samples = config["scenario"].get("export_samples")
        if export_samples is not None and export_samples > 0:
            sample_exporter = SampleExporter(
                self.scenario_output_dir, test_data.context, export_samples
            )
        else:
            sample_exporter = None

        for x, y in tqdm(test_data, desc="Attack"):
            with metrics.resource_context(
                name="Attack",
                profiler=config["metric"].get("profiler_type"),
                computational_resource_dict=metrics_logger.computational_resource_dict,
            ):

                if x.shape[0] != 1:
                    raise ValueError("D-APRICOT batch size must be set to 1")
                # (nb=1, num_cameras, h, w, c) --> (num_cameras, h, w, c)
                x = x[0]
                y_object, y_patch_metadata = y

                generate_kwargs = deepcopy(attack_config.get("generate_kwargs", {}))
                generate_kwargs["y_patch_metadata"] = y_patch_metadata
                y_target = label_targeter.generate(y_object)
                generate_kwargs["y_object"] = y_target

                x_adv = attack.generate(x=x, **generate_kwargs)

            # Ensure that input sample isn't overwritten by estimator
            x_adv.flags.writeable = False
            y_pred_adv = estimator.predict(x_adv)
            for img_idx in range(len(y_object)):
                y_i_target = y_target[img_idx]
                y_i_pred = y_pred_adv[img_idx]
                metrics_logger.update_task(
                    [y_i_target], [y_i_pred], adversarial=True, targeted=True
                )

            metrics_logger.update_perturbation(x, x_adv)
            if sample_exporter is not None:
                sample_exporter.export(x, x_adv, y_object, y_pred_adv)

        metrics_logger.log_task(adversarial=True, targeted=True)
        return metrics_logger.results()
