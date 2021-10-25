"""
CARLA object detection

Scenario Contributor: MITRE Corporation
"""

import logging

from armory.scenarios.scenario import Scenario
from armory.utils import metrics

logger = logging.getLogger(__name__)


class CarlaObjectDetectionTask(Scenario):
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

        # convert dict to List[dict] to comply with ART format
        y_object = [y_object]

        x.flags.writeable = False

        with metrics.resource_context(name="Inference", **self.profiler_kwargs):
            y_pred = self.model.predict(x, **self.predict_kwargs)
        self.metrics_logger.update_task(y_object, y_pred)
        self.y_pred = y_pred

    def run_attack(self):
        x, y = self.x, self.y
        y_object, y_patch_metadata = y

        # convert dict to List[dict] to comply with ART format
        y_object = [y_object]

        with metrics.resource_context(name="Attack", **self.profiler_kwargs):
            if self.use_label:
                y_target = [y_object]
            elif self.targeted:
                y_target = self.label_targeter.generate(y_object)
            else:
                y_target = None

            x_adv = self.attack.generate(
                x=x,
                y=y_target,
                y_patch_metadata=[y_patch_metadata],
                **self.generate_kwargs
            )

        # Ensure that input sample isn't overwritten by model
        x_adv.flags.writeable = False
        y_pred_adv = self.model.predict(x_adv, **self.predict_kwargs)
        self.metrics_logger.update_task(y_object, y_pred_adv, adversarial=True)
        if self.targeted:
            self.metrics_logger.update_task(
                y_target, y_pred_adv, adversarial=True, targeted=True
            )
        self.metrics_logger.update_perturbation(x, x_adv)

        # If using multimodal input, add a warning if depth channels are perturbed
        if x.shape[-1] == 6:
            if (x[..., 3:] != x_adv[..., 3:]).sum() > 0:
                logger.warning("Adversarial attack perturbed depth channels")

        if self.sample_exporter is not None:
            self.sample_exporter.export(x, x_adv, y, y_pred_adv)

        self.x_adv, self.y_target, self.y_pred_adv = x_adv, y_target, y_pred_adv
