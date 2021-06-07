"""
General audio classification scenario
"""

import logging

from armory.scenarios.scenario import Scenario

logger = logging.getLogger(__name__)


class AudioClassificationTask(Scenario):
    def load_dataset(self):
        if self.config["dataset"]["batch_size"] != 1:
            logger.warning("Evaluation batch_size != 1 may not be supported.")
        super().load_dataset()
