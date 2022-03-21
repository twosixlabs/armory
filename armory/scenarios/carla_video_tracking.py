"""
Classifier evaluation within ARMORY

Scenario Contributor: MITRE Corporation
"""


import numpy as np

from armory.scenarios.scenario import Scenario
from armory.utils import metrics
from armory.utils.export import VideoTrackingExporter


class CarlaVideoTracking(Scenario):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.skip_misclassified:
            raise ValueError(
                "skip_misclassified shouldn't be set for carla_video_tracking scenario"
            )

    def load_dataset(self):
        if self.config["dataset"]["batch_size"] != 1:
            raise ValueError("batch_size must be 1 for evaluation.")
        super().load_dataset(eval_split_default="dev")

    def run_benign(self):
        x, y = self.x, self.y
        y_object, y_patch_metadata = y
        y_init = np.expand_dims(y_object[0]["boxes"][0], axis=0)
        x.flags.writeable = False
        with metrics.resource_context(name="Inference", **self.profiler_kwargs):
            y_pred = self.model.predict(x, y_init=y_init, **self.predict_kwargs)
        self.metrics_logger.update_task(y_object, y_pred)
        self.y_pred = y_pred

    def run_attack(self):
        x, y = self.x, self.y
        self.y_object, self.y_patch_metadata = y
        y_init = np.expand_dims(self.y_object[0]["boxes"][0], axis=0)

        with metrics.resource_context(name="Attack", **self.profiler_kwargs):
            if self.use_label:
                y_target = self.y_object
            elif self.targeted:
                y_target = self.label_targeter.generate(self.y_object)
            else:
                y_target = None

            x_adv = self.attack.generate(
                x=x,
                y=y_target,
                y_patch_metadata=[self.y_patch_metadata],
                **self.generate_kwargs,
            )

        # Ensure that input sample isn't overwritten by model
        x_adv.flags.writeable = False

        y_pred_adv = self.model.predict(x_adv, y_init=y_init, **self.predict_kwargs)

        self.metrics_logger.update_task(self.y_object, y_pred_adv, adversarial=True)
        if self.targeted:
            self.metrics_logger.update_task(
                y_target, y_pred_adv, adversarial=True, targeted=True
            )
        self.metrics_logger.update_perturbation(x, x_adv)

        self.x_adv, self.y_target, self.y_pred_adv = x_adv, y_target, y_pred_adv

    def _load_sample_exporter(self):
        return VideoTrackingExporter(
            self.scenario_output_dir, frame_rate=self.test_dataset.context.frame_rate,
        )

    def export_samples(self):
        self._check_x("export_samples")
        self.sample_exporter.export(
            x=self.x,
            x_adv=self.x_adv,
            y=self.y_object,
            y_pred_clean=self.y_pred,
            y_pred_adv=self.y_pred_adv,
        )
