"""
General image recognition scenario for image classification and object detection.
"""

import numpy as np

from armory.instrument.export import ImageClassificationExporter
from armory.scenarios.scenario import Scenario


class ImageClassificationTask(Scenario):
    def load_attack(self):
        super().load_attack()
        # Temporary workaround for ART code requirement of ndarray mask
        if "mask" in self.generate_kwargs:
            self.generate_kwargs["mask"] = np.array(self.generate_kwargs["mask"])

    def _load_sample_exporter(self):
        return ImageClassificationExporter(self.export_dir)
