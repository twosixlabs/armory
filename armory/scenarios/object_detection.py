"""
General object detection scenario
"""

from armory.scenarios.image_classification import ImageClassificationTask
from armory.utils.export import SampleExporter


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

    def load_sample_exporter(self):
        export_samples = self.config["scenario"].get("export_samples")
        if export_samples is not None and export_samples > 0:
            # Sample exporting needs access to benign predictions to include bounding boxes
            if self.skip_benign:
                raise ValueError(
                    "--skip-benign should not be set for object_detection scenario if export_samples is enabled."
                )
        super().load_sample_exporter()

    def export_samples(self):
        self.sample_exporter.export(
            self.x, self.x_adv, self.y, self.y_pred_adv, self.y_pred, plot_bboxes=True
        )
