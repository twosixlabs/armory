"""
General audio classification scenario
"""

from armory.instrument.export import AudioExporter
from armory.logs import log
from armory.scenarios.scenario import Scenario


class AudioClassificationTask(Scenario):
    def load_dataset(self):
        if self.config["dataset"]["batch_size"] != 1:
            log.warning("Evaluation batch_size != 1 may not be supported.")
        super().load_dataset()

    def _load_sample_exporter(self):
        return AudioExporter(
            self.export_dir,
            self.test_dataset.context.sample_rate,
        )
