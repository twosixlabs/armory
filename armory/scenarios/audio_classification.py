"""
General audio classification scenario
"""

from armory.scenarios.scenario import Scenario
from armory.logs import log
from armory.utils.export import AudioExporter


class AudioClassificationTask(Scenario):
    def load_dataset(self):
        if self.config["dataset"]["batch_size"] != 1:
            log.warning("Evaluation batch_size != 1 may not be supported.")
        super().load_dataset()

    def _load_sample_exporter(self):
        return AudioExporter(
            self.scenario_output_dir,
            self.test_dataset.context.sample_rate,
        )
