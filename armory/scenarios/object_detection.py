"""
General object detection scenario
"""

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
