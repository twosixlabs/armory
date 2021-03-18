"""
Multimodal image classification, currently designed for So2Sat dataset
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


class So2SatClassification(Scenario):
    def __init__(self, **kwargs):
        if "attack_modality" not in kwargs.keys():
            raise ValueError("`attack_modality` must be defined for So2Sat scenario")
        if kwargs["attack_modality"] is None or kwargs[
            "attack_modality"
        ].lower() not in ("sar", "eo", "both",):
            raise ValueError(
                f"Multimodal scenario requires attack_modality parameter in {'SAR', 'EO', 'Both'}"
            )
        self.attack_modality = kwargs["attack_modality"].lower()
        super().__init__()

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

        attack_config = config["attack"]
        attack_channels_mask = attack_config.get("generate_kwargs", {}).get("mask")

        if attack_channels_mask is None:
            if self.attack_modality == "sar":
                logger.info("No mask configured. Attacking all SAR channels")
                attack_channels_mask = np.concatenate(
                    (np.ones(4, dtype=np.float32), np.zeros(10, dtype=np.float32)),
                    axis=0,
                )
            elif self.attack_modality == "eo":
                logger.info("No mask configured. Attacking all EO channels")
                attack_channels_mask = np.concatenate(
                    (np.zeros(4, dtype=np.float32), np.ones(10, dtype=np.float32)),
                    axis=0,
                )
            elif self.attack_modality == "both":
                logger.info("No mask configured. Attacking all SAR and EO channels")
                attack_channels_mask = np.ones(14, dtype=np.float32)

        else:
            assert isinstance(
                attack_channels_mask, list
            ), "Mask is specified, but incorrect format. Expected list"
            attack_channels_mask = np.array(attack_channels_mask)
            where_mask = np.where(attack_channels_mask)[0]
            if self.attack_modality == "sar":
                assert np.all(
                    np.logical_and(where_mask >= 0, where_mask < 4)
                ), "Selected SAR-only attack modality, but specify non-SAR channels"
            elif self.attack_modality == "eo":
                assert np.all(
                    np.logical_and(where_mask >= 4, where_mask < 14)
                ), "Selected EO-only attack modality, but specify non-EO channels"
            elif self.attack_modality == "both":
                assert np.all(
                    np.logical_and(where_mask >= 0, where_mask < 14)
                ), "Selected channels are out-of-bounds"
        assert (
            len(attack_channels_mask) == 14
        ), f"Expected channel mask of length 14, found length {len(attack_channels_mask)}"
        assert np.all(
            np.logical_or(attack_channels_mask == 0, attack_channels_mask == 1)
        ), "Expected binary attack channel mask, but found values outside {0,1}"

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

        attack_type = attack_config.get("type")
        targeted = bool(attack_config.get("kwargs", {}).get("targeted"))

        performance_metrics = deepcopy(config["metric"])
        performance_metrics.pop("perturbation")
        performance_logger = metrics.MetricsLogger.from_config(
            performance_metrics,
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
                    computational_resource_dict=performance_logger.computational_resource_dict,
                ):
                    y_pred = estimator.predict(x)
                performance_logger.update_task(y, y_pred)
            performance_logger.log_task()

        if skip_attack:
            logger.info("Skipping attack generation...")
            return performance_logger.results()

        # Evaluate the ART estimator on adversarial test examples
        logger.info("Generating or loading / testing adversarial examples...")

        if skip_misclassified:
            acc_task_idx = [i.name for i in performance_logger.tasks].index(
                "categorical_accuracy"
            )
            benign_acc = performance_logger.tasks[acc_task_idx].values()

        perturbation_metrics = deepcopy(config["metric"])
        perturbation_metrics.pop("task")
        if self.attack_modality in ("sar", "both"):
            sar_perturbation_logger = metrics.MetricsLogger.from_config(
                perturbation_metrics,
                skip_benign=True,
                skip_attack=False,
                targeted=targeted,
            )
        else:
            sar_perturbation_logger = None

        if self.attack_modality in ("eo", "both"):
            eo_perturbation_logger = metrics.MetricsLogger.from_config(
                perturbation_metrics,
                skip_benign=True,
                skip_attack=False,
                targeted=targeted,
            )
        else:
            eo_perturbation_logger = None

        if targeted and attack_config.get("use_label"):
            raise ValueError("Targeted attacks cannot have 'use_label'")
        if attack_type == "preloaded":
            test_data = load_adversarial_dataset(
                attack_config,
                epochs=1,
                split="adversarial",
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
                computational_resource_dict=performance_logger.computational_resource_dict,
            ):
                if attack_type == "preloaded":
                    logger.warning(
                        "Specified preloaded attack. Ignoring `attack_modality` parameter"
                    )
                    if len(x) == 2:
                        x, x_adv = x
                    else:
                        x_adv = x
                    if targeted:
                        y, y_target = y
                else:
                    generate_kwargs = deepcopy(attack_config.get("generate_kwargs", {}))
                    generate_kwargs["mask"] = attack_channels_mask
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
            performance_logger.update_task(y, y_pred_adv, adversarial=True)
            if targeted:
                performance_logger.update_task(
                    y_target, y_pred_adv, adversarial=True, targeted=True
                )

            # Update perturbation metrics for SAR/EO separately
            x_sar = np.stack(
                (x[..., 0] + 1j * x[..., 1], x[..., 2] + 1j * x[..., 3]), axis=3
            )
            x_adv_sar = np.stack(
                (
                    x_adv[..., 0] + 1j * x_adv[..., 1],
                    x_adv[..., 2] + 1j * x_adv[..., 3],
                ),
                axis=3,
            )
            x_eo = x[..., 4:]
            x_adv_eo = x_adv[..., 4:]
            if sar_perturbation_logger is not None:
                sar_perturbation_logger.update_perturbation(x_sar, x_adv_sar)
            if eo_perturbation_logger is not None:
                eo_perturbation_logger.update_perturbation(x_eo, x_adv_eo)

            if sample_exporter is not None:
                sample_exporter.export(x, x_adv, y, y_pred_adv)

        performance_logger.log_task(adversarial=True)
        if targeted:
            performance_logger.log_task(adversarial=True, targeted=True)

        # Merge performance, SAR, EO results
        combined_results = performance_logger.results()
        if sar_perturbation_logger is not None:
            combined_results.update(
                {f"sar_{k}": v for k, v in sar_perturbation_logger.results().items()}
            )
        if eo_perturbation_logger is not None:
            combined_results.update(
                {f"eo_{k}": v for k, v in eo_perturbation_logger.results().items()}
            )
        return combined_results
