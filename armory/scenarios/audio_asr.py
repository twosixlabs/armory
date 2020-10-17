"""
Automatic speech recognition scenario
"""

import logging
from typing import Optional

from tqdm import tqdm

from armory.utils.config_loading import (
    load_dataset,
    load_model,
    load_attack,
    load_adversarial_dataset,
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
        model_config = config["model"]
        classifier, _ = load_model(model_config)

        if model_config["fit"]:
            raise NotImplementedError("Model fit not yet implemented for ASR")

        predict_kwargs = config["model"].get("predict_kwargs", {})
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
                num_batches=num_eval_batches,
                shuffle_files=False,
            )
            logger.info("Running inference on benign examples...")
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
        if (config.get("adhoc") or {}).get("skip_adversarial"):
            logger.info("Skipping adversarial classification...")
        else:
            # Evaluate the ART classifier on adversarial test examples
            logger.info("Generating or loading / testing adversarial examples...")

            attack_config = config["attack"]
            attack_type = attack_config.get("type")
            if attack_type == "preloaded":
                test_data = load_adversarial_dataset(
                    attack_config,
                    epochs=1,
                    split_type="adversarial",
                    num_batches=num_eval_batches,
                    shuffle_files=False,
                )
            else:
                attack = load_attack(attack_config, classifier)
                test_data = load_dataset(
                    config["dataset"],
                    epochs=1,
                    split_type="test",
                    num_batches=num_eval_batches,
                    shuffle_files=False,
                )
            for x, y in tqdm(test_data, desc="Attack"):
                with metrics.resource_context(
                    name="Attack",
                    profiler=config["metric"].get("profiler_type"),
                    computational_resource_dict=metrics_logger.computational_resource_dict,
                ):
                    if attack_type == "preloaded":
                        x, x_adv = x
                        y, y_target = y
                    elif attack_config.get("use_label"):
                        x_adv = attack.generate(x=x, y=y)
                    else:
                        x_adv = attack.generate(x=x, y=["TEST STRING"])

                # Ensure that input sample isn't overwritten by classifier
                x_adv.flags.writeable = False
                y_pred_adv = classifier.predict(x_adv, **predict_kwargs)
                metrics_logger.update_task(y, y_pred_adv, adversarial=True)
                metrics_logger.update_perturbation(x, x_adv)
            metrics_logger.log_task(adversarial=True, targeted=True)
        return metrics_logger.results()
