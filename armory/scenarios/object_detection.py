"""
General object detection scenario
"""

from armory.instrument.export import (
    CocoBoxFormatMeter,
    ExportMeter,
    ObjectDetectionExporter,
)
from armory.scenarios.image_classification import ImageClassificationTask


class ObjectDetectionTask(ImageClassificationTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.skip_misclassified:
            raise ValueError(
                "skip_misclassified shouldn't be set for object detection scenario"
            )

    def fit(self, train_split_default="train"):
        raise NotImplementedError(
            "Training has not yet been implemented for object detectors"
        )

    def load_export_meters(self):
        super().load_export_meters()
        self.sample_exporter_with_boxes = self._load_sample_exporter_with_boxes()
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
            if self.skip_attack:
                break

        coco_box_format_meter = CocoBoxFormatMeter(
            "coco_box_format_meter",
            self.export_dir,
            y_probe="scenario.y",
            y_pred_clean_probe="scenario.y_pred" if not self.skip_benign else None,
            y_pred_adv_probe="scenario.y_pred_adv" if not self.skip_attack else None,
            max_batches=self.num_export_batches,
        )
        self.hub.connect_meter(coco_box_format_meter, use_default_writers=False)

    def _load_sample_exporter(self):
        return ObjectDetectionExporter(self.export_dir)

    def _load_sample_exporter_with_boxes(self):
        return ObjectDetectionExporter(
            self.export_dir, default_export_kwargs={"with_boxes": True}
        )
