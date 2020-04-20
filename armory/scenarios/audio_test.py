"""
Test scenario for adversarial librispeech
"""

import logging

from tqdm import tqdm

from armory.utils.config_loading import (
    load_dataset,
    load_model,
    load_attack,
    load_defense,
)
from armory.utils import metrics
from armory.scenarios.base import Scenario
from armory import paths
logger = logging.getLogger(__name__)


class AudioTest(Scenario):
    def _evaluate(self, config: dict) -> dict:
        """
        Evaluate the config and return a results dict
        """
        model_config = config["model"]
        classifier, preprocessing_fn = load_model(model_config)
        classifier.set_learning_phase(False)
        # Evaluate the ART classifier on benign test examples
        logger.info(f"Loading dataset {config['dataset']['name']}...")
        test_data_generator = load_dataset(
            config["dataset"],
            epochs=1,
            split_type="test",
            preprocessing_fn=preprocessing_fn,
        )
        logger.info("Running inference on benign examples...")
        metrics_logger = metrics.MetricsLogger.from_config(config["metric"])

        for x, y in tqdm(test_data_generator, desc="Benign"):
            y_pred = classifier.predict(x)
            metrics_logger.update_task(y, y_pred)
        metrics_logger.log_task()
        return metrics_logger.results()
