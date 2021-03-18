"""
General object detection scenario
"""

from typing import Optional

from armory.scenarios.image_classification import ImageClassificationTask


class ObjectDetectionTask(ImageClassificationTask):
    def _evaluate(
        self,
        config: dict,
        num_eval_batches: Optional[int],
        skip_benign: Optional[bool],
        skip_attack: Optional[bool],
        skip_misclassified: Optional[bool],
    ):
        if skip_misclassified:
            raise ValueError(
                "skip_misclassified shouldn't be set for object detection scenario"
            )

        return super()._evaluate(
            config=config,
            num_eval_batches=num_eval_batches,
            skip_benign=skip_benign,
            skip_attack=skip_attack,
            skip_misclassified=None,
        )
