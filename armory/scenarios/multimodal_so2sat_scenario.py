"""
Multimodal image classification, currently designed for So2Sat dataset
"""

import copy
import logging

import numpy as np

from armory.utils import metrics
from armory.scenarios.scenario import Scenario

logger = logging.getLogger(__name__)


class So2SatClassification(Scenario):
    def __init__(self, *args, **kwargs):
        if "attack_modality" not in kwargs.keys():
            raise ValueError("`attack_modality` must be defined for So2Sat scenario")
        attack_modality = (kwargs.pop("attack_modality") or "").lower()
        if attack_modality not in ("sar", "eo", "both"):
            raise ValueError(
                f"Multimodal scenario requires attack_modality parameter in {'SAR', 'EO', 'Both'}"
            )
        self.attack_modality = attack_modality
        super().__init__(*args, **kwargs)

    def load_attack(self):
        attack_config = self.config["attack"]
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
        super().load_attack()
        self.generate_kwargs["mask"] = attack_channels_mask

    def load_metrics(self):
        super().load_metrics()

        # Overwrite standard metrics_logger
        metrics_config = self.config["metric"]
        performance_metrics = copy.deepcopy(metrics_config)
        performance_metrics.pop("perturbation")
        metrics_logger = metrics.MetricsLogger.from_config(
            performance_metrics,
            skip_benign=self.skip_benign,
            skip_attack=self.skip_attack,
            targeted=self.targeted,
        )
        self.profiler_kwargs[
            "computational_resource_dict"
        ] = metrics_logger.computational_resource_dict

        perturbation_metrics = copy.deepcopy(self.config["metric"])
        perturbation_metrics.pop("task")
        if self.attack_modality in ("sar", "both"):
            sar_perturbation_logger = metrics.MetricsLogger.from_config(
                perturbation_metrics,
                skip_benign=True,
                skip_attack=False,
                targeted=self.targeted,
            )
        else:
            sar_perturbation_logger = None

        if self.attack_modality in ("eo", "both"):
            eo_perturbation_logger = metrics.MetricsLogger.from_config(
                perturbation_metrics,
                skip_benign=True,
                skip_attack=False,
                targeted=self.targeted,
            )
        else:
            eo_perturbation_logger = None

        self.metrics_logger = metrics_logger
        self.sar_perturbation_logger = sar_perturbation_logger
        self.eo_perturbation_logger = eo_perturbation_logger

    def run_attack(self):
        x, y, y_pred = self.x, self.y, self.y_pred

        with metrics.resource_context(name="Attack", **self.profiler_kwargs):
            if self.attack_type == "preloaded":
                logger.warning(
                    "Specified preloaded attack. Ignoring `attack_modality` parameter"
                )
                if len(x) == 2:
                    x, x_adv = x
                else:
                    x_adv = x
                if self.targeted:
                    y, y_target = y
                else:
                    y_target = None

                misclassified = False
            else:
                if self.use_label:
                    y_target = y
                elif self.targeted:
                    y_target = self.label_targeter.generate(y)
                elif self.skip_benign:
                    y_target = None  # most attacks will call self.model.predict(x)
                else:
                    y_target = y_pred

                if self.skip_misclassified:
                    if self.targeted:
                        misclassified = all(
                            metrics.categorical_accuracy(y_target, y_pred)
                        )
                    else:
                        misclassified = not any(metrics.categorical_accuracy(y, y_pred))
                else:
                    misclassified = False

                if misclassified:
                    x_adv = x
                else:
                    x_adv = self.attack.generate(
                        x=x, y=y_target, **self.generate_kwargs
                    )

        if misclassified:
            y_pred_adv = y_pred
        else:
            # Ensure that input sample isn't overwritten by model
            x_adv.flags.writeable = False
            y_pred_adv = self.model.predict(x_adv, **self.predict_kwargs)

        self.metrics_logger.update_task(y, y_pred_adv, adversarial=True)
        if self.targeted:
            self.metrics_logger.update_task(
                y_target, y_pred_adv, adversarial=True, targeted=True
            )

        # Update perturbation metrics for SAR/EO separately
        x_sar = np.stack(
            (x[..., 0] + 1j * x[..., 1], x[..., 2] + 1j * x[..., 3]), axis=3
        )
        x_adv_sar = np.stack(
            (x_adv[..., 0] + 1j * x_adv[..., 1], x_adv[..., 2] + 1j * x_adv[..., 3],),
            axis=3,
        )
        x_eo = x[..., 4:]
        x_adv_eo = x_adv[..., 4:]
        if self.sar_perturbation_logger is not None:
            self.sar_perturbation_logger.update_perturbation(x_sar, x_adv_sar)
        if self.eo_perturbation_logger is not None:
            self.eo_perturbation_logger.update_perturbation(x_eo, x_adv_eo)

        if self.sample_exporter is not None:
            self.sample_exporter.export(x, x_adv, y, y_pred_adv)

        self.x_adv, self.y_target, self.y_pred_adv = x_adv, y_target, y_pred_adv

    def finalize_results(self):
        metrics_logger = self.metrics_logger
        metrics_logger.log_task()
        metrics_logger.log_task(adversarial=True)
        if self.targeted:
            metrics_logger.log_task(adversarial=True, targeted=True)
        self.results = metrics_logger.results()
        metrics_logger.log_task(adversarial=True)
        if self.targeted:
            metrics_logger.log_task(adversarial=True, targeted=True)

        # Merge performance, SAR, EO results
        combined_results = metrics_logger.results()
        if self.sar_perturbation_logger is not None:
            combined_results.update(
                {
                    f"sar_{k}": v
                    for k, v in self.sar_perturbation_logger.results().items()
                }
            )
        if self.eo_perturbation_logger is not None:
            combined_results.update(
                {f"eo_{k}": v for k, v in self.eo_perturbation_logger.results().items()}
            )
        self.results = combined_results
