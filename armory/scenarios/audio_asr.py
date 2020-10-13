"""
Automatic speech recognition scenario
"""

import logging
from typing import Optional

from tqdm import tqdm

from armory.utils.config_loading import (
    load_dataset,
    load_model,
)
from armory.utils import metrics
from armory.scenarios.base import Scenario

logger = logging.getLogger(__name__)


class AutomaticSpeechRecognition(Scenario):
    def _evaluate(
        self, config: dict, num_eval_batches: Optional[int], skip_benign: Optional[bool]
    ) -> dict:
        """
        Evaluate the config and return a results dict
        """
        skip_benign = False
        model_config = config["model"]
        classifier, preprocessing_fn = load_model(model_config)
        if isinstance(preprocessing_fn, tuple):
            fit_preprocessing_fn, predict_preprocessing_fn = preprocessing_fn
        else:
            fit_preprocessing_fn = (
                predict_preprocessing_fn
            ) = preprocessing_fn  # noqa: F841

        metrics_logger = metrics.MetricsLogger.from_config(
            config["metric"], skip_benign=skip_benign
        )
        if skip_benign:
            logger.info("Skipping benign classification...")
        else:
            # Evaluate the ART classifier on benign test examples
            logger.info(f"Loading test dataset {config['dataset']['name']}...")
            test_data = load_dataset(
                config["dataset"],
                epochs=1,
                split_type="test",
                preprocessing_fn=predict_preprocessing_fn,
                num_batches=num_eval_batches,
                shuffle_files=False,
            )
            logger.info("Running inference on benign examples...")
            predict_kwargs = config["adhoc"]["predict_kwargs"]
            for x, y in tqdm(test_data, desc="Benign"):
                # Ensure that input sample isn't overwritten by classifier
                x.flags.writeable = False
                with metrics.resource_context(
                    name="Inference",
                    profiler=config["metric"].get("profiler_type"),
                    computational_resource_dict=metrics_logger.computational_resource_dict,
                ):
                    y_pred = classifier.predict(x, **predict_kwargs)
                metrics_logger.update_task(y, y_pred)
            metrics_logger.log_task()
        # Imperceptible attack still WIP
        return metrics_logger.results()
