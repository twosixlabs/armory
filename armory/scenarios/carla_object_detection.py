"""
CARLA object detection

Scenario Contributor: MITRE Corporation
"""

from armory.instrument.export import ObjectDetectionExporter
from armory.logs import log
from armory.scenarios.object_detection import ObjectDetectionTask


class CarlaObjectDetectionTask(ObjectDetectionTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.skip_benign:
            raise ValueError(
                "skip_benign shouldn't be set for carla_object_detection scenario, as "
                "adversarial predictions are measured against benign predictions"
            )

    def load_dataset(self):
        if self.config["dataset"]["batch_size"] != 1:
            raise ValueError("batch_size must be 1 for evaluation.")
        super().load_dataset(eval_split_default="dev")

    def next(self):
        super().next()
        # The CARLA dev and test sets (as opposed to train/val) contain green-screens
        # and thus have a tuple of two types of labels that we separate here
        if isinstance(self.y, tuple):
            self.y, self.y_patch_metadata = [[y_i] for y_i in self.y]
            self.probe.update(y=self.y, y_patch_metadata=self.y_patch_metadata)

    def run_attack(self):
        self._check_x("run_attack")
        if not hasattr(self, "y_patch_metadata"):
            raise AttributeError(
                "y_patch_metadata attribute does not exist. Please set --skip-attack if using "
                "CARLA train set"
            )
        self.hub.set_context(stage="attack")
        x, y = self.x, self.y

        with self.profiler.measure("Attack"):
            if self.use_label:
                y_target = y
            elif self.targeted:
                y_target = self.label_targeter.generate(y)
            else:
                y_target = None

            x_adv = self.attack.generate(
                x=x,
                y=y_target,
                y_patch_metadata=self.y_patch_metadata,
                **self.generate_kwargs,
            )

        # Ensure that input sample isn't overwritten by model
        self.hub.set_context(stage="adversarial")
        x_adv.flags.writeable = False
        y_pred_adv = self.model.predict(x_adv, **self.predict_kwargs)

        self.probe.update(x_adv=x_adv, y_pred_adv=y_pred_adv)
        if self.targeted:
            self.probe.update(y_target=y_target)

        # If using multimodal input, add a warning if depth channels are perturbed
        if x.shape[-1] == 6:
            if (x[..., 3:] != x_adv[..., 3:]).sum() > 0:
                log.warning("Adversarial attack perturbed depth channels")

        self.x_adv, self.y_target, self.y_pred_adv = x_adv, y_target, y_pred_adv

    def load_metrics(self):
        super().load_metrics()
        # measure adversarial results using benign predictions as labels
        self.metrics_logger.add_tasks_wrt_benign_predictions()

    def _load_sample_exporter_with_boxes(self):
        return ObjectDetectionExporter(
            self.export_dir,
            default_export_kwargs={"with_boxes": True, "classes_to_skip": [4]},
        )
