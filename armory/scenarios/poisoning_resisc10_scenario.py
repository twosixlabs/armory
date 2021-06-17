"""
Classifier evaluation within ARMORY

Scenario Contributor: MITRE Corporation
"""

import logging

from armory.scenarios.poison import Poison
from armory.utils import config_loading

logger = logging.getLogger(__name__)


class RESISC10(Poison):
    def set_dataset_kwargs(self):
        super().set_dataset_kwargs()
        dataset_config = self.config["dataset"]
        self.validation_split = dataset_config.get("eval_split", "validation")

    def load_model(self, defended=True):
        super().load_model(defended=defended)
        defense_config = self.config.get("defense") or {}
        if "data_augmentation" in defense_config:
            for data_aug_config in defense_config["data_augmentation"].values():
                estimator = config_loading.load_defense_internal(
                    data_aug_config, self.estimator
                )
        logger.info(
            f"estimator.preprocessing_defences: {estimator.preprocessing_defences}"
        )
        self.estimator = estimator
