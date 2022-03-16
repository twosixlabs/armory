"""
General audio classification scenario
"""

from armory.scenarios.scenario import Scenario
from armory.logs import log


class AudioClassificationTask(Scenario):
    def load_dataset(self):
        if self.config["dataset"]["batch_size"] != 1:
            log.warning("Evaluation batch_size != 1 may not be supported.")
        super().load_dataset()
