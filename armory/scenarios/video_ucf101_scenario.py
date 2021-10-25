"""
Classifier evaluation within ARMORY

Scenario Contributor: MITRE Corporation
"""

import logging

from armory.scenarios.scenario import Scenario

logger = logging.getLogger(__name__)


class Ucf101(Scenario):
    def load_dataset(self):
        if self.config["dataset"]["batch_size"] != 1:
            raise ValueError(
                "batch_size must be 1 for evaluation, due to variable length inputs.\n"
                "    If training, set config['model']['fit_kwargs']['fit_batch_size']"
            )
        super().load_dataset()
