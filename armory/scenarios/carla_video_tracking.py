"""
Classifier evaluation within ARMORY

Scenario Contributor: MITRE Corporation
"""

import logging

import numpy as np

from armory.scenarios.scenario import Scenario
from armory.utils import metrics

logger = logging.getLogger(__name__)


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
        y_init = np.expand_dims(y_object[0], axis=0)
        x.flags.writeable = False
        with metrics.resource_context(name="Inference", **self.profiler_kwargs):
            y_pred = self.model.predict(x, y_init=y_init, **self.predict_kwargs)
        self.metrics_logger.update_task(y_object, y_pred)
        self.y_pred = y_pred
