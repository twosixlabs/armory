"""
Multimodal image classification, currently designed for So2Sat dataset
"""

import numpy as np

from armory import metrics
from armory.instrument import Meter
from armory.instrument.export import So2SatExporter
from armory.logs import log
from armory.scenarios.scenario import Scenario


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
        self.perturbation_metrics = self.config["metric"].pop("perturbation")
        self.config["metric"]["perturbation"] = None
        if self.perturbation_metrics is not None:
            if isinstance(self.perturbation_metrics, str):
                self.perturbation_metrics = [self.perturbation_metrics]

    def load_attack(self):
        attack_config = self.config["attack"]
        attack_channels_mask = attack_config.get("generate_kwargs", {}).get("mask")
        if attack_channels_mask is None:
            if self.attack_modality == "sar":
                log.info("No mask configured. Attacking all SAR channels")
                attack_channels_mask = np.concatenate(
                    (np.ones(4, dtype=np.float32), np.zeros(10, dtype=np.float32)),
                    axis=0,
                )
            elif self.attack_modality == "eo":
                log.info("No mask configured. Attacking all EO channels")
                attack_channels_mask = np.concatenate(
                    (np.zeros(4, dtype=np.float32), np.ones(10, dtype=np.float32)),
                    axis=0,
                )
            elif self.attack_modality == "both":
                log.info("No mask configured. Attacking all SAR and EO channels")
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
        if not self.perturbation_metrics:
            return

        if self.attack_modality == "both":
            modes = ["sar", "eo"]
        else:
            modes = [self.attack_modality]

        # Generate metrics for perturbation on eo and sar
        if self.config["metric"].get("means"):
            final = np.mean
        else:
            final = None
        for mode in modes:
            for name in self.perturbation_metrics:
                metric = metrics.get(name)
                m = Meter(
                    f"{mode}_perturbation_{name}",
                    metric,
                    f"scenario.x_{mode}",
                    f"scenario.x_adv_{mode}",
                    final=final,
                    final_name=f"{mode}_perturbation_mean_{name}",
                    record_final_only=not bool(
                        self.config["metric"].get("record_metric_per_sample")
                    ),
                )
                self.hub.connect_meter(m)

    def run_benign(self):
        super().run_benign()
        x = self.x
        x_sar = np.stack(
            (x[..., 0] + 1j * x[..., 1], x[..., 2] + 1j * x[..., 3]), axis=3
        )
        x_eo = x[..., 4:]
        self.probe.update(x_sar=x_sar, x_eo=x_eo)
        self.x_sar, self.x_eo = x_sar, x_eo

    def run_attack(self):
        self._check_x("run_attack")
        self.hub.set_context(stage="attack")
        x, y, y_pred = self.x, self.y, self.y_pred

        with self.profiler.measure("Attack"):
            if self.attack_type == "preloaded":
                log.warning(
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
                            metrics.task.batch.categorical_accuracy(y_target, y_pred)
                        )
                    else:
                        misclassified = not any(
                            metrics.task.batch.categorical_accuracy(y, y_pred)
                        )
                else:
                    misclassified = False

                if misclassified:
                    x_adv = x
                else:
                    x_adv = self.attack.generate(
                        x=x, y=y_target, **self.generate_kwargs
                    )

        self.hub.set_context(stage="adversarial")
        if misclassified:
            y_pred_adv = y_pred
        else:
            # Ensure that input sample isn't overwritten by model
            x_adv.flags.writeable = False
            y_pred_adv = self.model.predict(x_adv, **self.predict_kwargs)

        self.probe.update(x_adv=x_adv, y_pred_adv=y_pred_adv)
        if self.targeted:
            self.probe.update(y_target=y_target)

        # Update perturbation metrics for SAR/EO separately
        if self.attack_modality in ("sar", "both"):
            x_adv_sar = np.stack(
                (
                    x_adv[..., 0] + 1j * x_adv[..., 1],
                    x_adv[..., 2] + 1j * x_adv[..., 3],
                ),
                axis=3,
            )
            self.probe.update(x_adv_sar=x_adv_sar)
        if self.attack_modality in ("eo", "both"):
            x_adv_eo = x_adv[..., 4:]
            self.probe.update(x_adv_eo=x_adv_eo)

        self.x_adv, self.y_target, self.y_pred_adv = x_adv, y_target, y_pred_adv

    def _load_sample_exporter(self):
        return So2SatExporter(self.export_dir)
