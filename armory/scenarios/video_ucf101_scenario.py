"""
Classifier evaluation within ARMORY

Scenario Contributor: MITRE Corporation
"""

from armory.scenarios.scenario import Scenario
from armory.instrument.export import VideoClassificationExporter
DEFAULT_FRAME_RATE = 10

class Ucf101(Scenario):
    def load_test_dataset(self):
        if self.config["dataset"].get("test").get("batch_size") != 1:
            raise ValueError(
                "batch_size must be 1 for evaluation, due to variable length inputs.\n"
                "    If training, set config['model']['fit_kwargs']['fit_batch_size']"
            )
        super().load_test_dataset()

    def _load_sample_exporter(self):
        return VideoClassificationExporter(
            self.export_dir,
            frame_rate=self.test_dataset.metadata.get("frame_rate", DEFAULT_FRAME_RATE),
        )
