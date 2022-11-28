"""
General audio classification scenario
"""

from armory.scenarios.scenario import Scenario
from armory.logs import log
from armory.instrument.export import AudioExporter


class AudioClassificationTask(Scenario):
    def load_test_dataset(self):
        if self.config["dataset"].get("test").get("batch_size") != 1:
            log.warning("Evaluation batch_size != 1 may not be supported.")
        super().load_test_dataset()

    def _load_sample_exporter(self):
        return AudioExporter(
            self.export_dir,
            self.test_dataset.context.sample_rate,
        )
