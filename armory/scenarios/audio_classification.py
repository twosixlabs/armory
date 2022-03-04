"""
General audio classification scenario
"""

import logging

from armory.scenarios.scenario import Scenario
from armory.utils.export import AudioExporter

logger = logging.getLogger(__name__)


class AudioClassificationTask(Scenario):
    def load_dataset(self):
        if self.config["dataset"]["batch_size"] != 1:
            logger.warning("Evaluation batch_size != 1 may not be supported.")
        super().load_dataset()

    def _load_sample_exporter(self):
        return AudioExporter(
            self.scenario_output_dir,
            self.num_export_samples,
            self.test_dataset.context.sample_rate,
        )
