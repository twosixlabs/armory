"""
General object detection scenario
"""

from armory.scenarios.image_classification import ImageClassificationTask
from armory.utils.export import ObjectDetectionExporter, ExportMeter


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
        export_samples = self.config["scenario"].get("export_samples")
        if export_samples is not None and export_samples > 0:
            # Sample exporting needs access to benign predictions to include bounding boxes
            if self.skip_benign:
                raise ValueError(
                    "--skip-benign should not be set for object_detection scenario if export_batches is enabled."
                )

        num_export_batches = self.config["scenario"].get("export_batches", 0)
        if num_export_batches is True:
            num_export_batches = len(self.test_dataset)
        self.num_export_batches = num_export_batches
        self.sample_exporter = self._load_sample_exporter()

        for probe_data, probe_pred in [("x", "y_pred"), ("x_adv", "y_pred_adv")]:
            export_with_boxes_meter = ExportMeter(
                f"{probe_data}_exporter",
                self.sample_exporter,
                f"scenario.{probe_data}",
                "scenario.y",
                f"scenario.{probe_pred}",
                max_batches=self.num_export_batches,
            )
            self.hub.connect_meter(export_with_boxes_meter, use_default_writers=False)

    def _load_sample_exporter(self):
        default_export_kwargs = {"with_boxes": True}
        return ObjectDetectionExporter(
            self.scenario_output_dir, default_export_kwargs=default_export_kwargs
        )
