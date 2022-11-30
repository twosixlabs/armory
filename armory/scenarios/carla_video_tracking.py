"""
Classifier evaluation within ARMORY

Scenario Contributor: MITRE Corporation
"""


import numpy as np

from armory.instrument.export import ExportMeter, VideoTrackingExporter
from armory.scenarios.scenario import Scenario


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

    def next(self):
        super().next()
        self.y, self.y_patch_metadata = self.y
        self.probe.update(y=self.y, y_patch_metadata=self.y_patch_metadata)

    def run_benign(self):
        self._check_x("run_benign")
        self.hub.set_context(stage="benign")
        x, y = self.x, self.y
        y_init = np.expand_dims(y[0]["boxes"][0], axis=0)
        x.flags.writeable = False
        with self.profiler.measure("Inference"):
            y_pred = self.model.predict(x, y_init=y_init, **self.predict_kwargs)
        self.y_pred = y_pred
        self.probe.update(y_pred=y_pred)

    def run_attack(self):
        self._check_x("run_attack")
        self.hub.set_context(stage="attack")
        x, y = self.x, self.y
        y_init = np.expand_dims(y[0]["boxes"][0], axis=0)

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
                y_patch_metadata=[self.y_patch_metadata],
                **self.generate_kwargs,
            )

        self.hub.set_context(stage="adversarial")
        # Ensure that input sample isn't overwritten by model
        x_adv.flags.writeable = False

        y_pred_adv = self.model.predict(x_adv, y_init=y_init, **self.predict_kwargs)

        self.probe.update(x_adv=x_adv, y_pred_adv=y_pred_adv)
        if self.targeted:
            self.probe.update(y_target=y_target)

        self.x_adv, self.y_target, self.y_pred_adv = x_adv, y_target, y_pred_adv

    def _load_sample_exporter(self):
        return VideoTrackingExporter(
            self.export_dir,
            frame_rate=self.test_dataset.context.frame_rate,
        )

    def load_export_meters(self):
        # Load default export meters
        super().load_export_meters()

        # Add export meters that export examples with boxes overlaid
        self.sample_exporter_with_boxes = VideoTrackingExporter(
            self.export_dir,
            frame_rate=self.test_dataset.context.frame_rate,
            default_export_kwargs={"with_boxes": True},
        )
        for probe_data, probe_pred in [("x", "y_pred"), ("x_adv", "y_pred_adv")]:
            export_with_boxes_meter = ExportMeter(
                f"{probe_data}_with_boxes_exporter",
                self.sample_exporter_with_boxes,
                f"scenario.{probe_data}",
                y_probe="scenario.y",
                y_pred_probe=f"scenario.{probe_pred}",
                max_batches=self.num_export_batches,
            )
            self.hub.connect_meter(export_with_boxes_meter, use_default_writers=False)
