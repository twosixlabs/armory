"""
General audio classification scenario
"""

import logging
from typing import Optional

from tqdm import tqdm

from armory.utils import config_loading  # Should be part of scenarios, not utils?

from armory.utils import metrics
from armory.scenarios.base import Scenario
from armory.utils.export import SampleExporter

logger = logging.getLogger(__name__)


class AudioClassificationUpdatedTask(Scenario):
    def _evaluate(
        self,
        config: dict,
        num_eval_batches: Optional[int],
        skip_benign: Optional[bool],
        skip_attack: Optional[bool],
        skip_misclassified: Optional[bool],
    ) -> dict:
        """
        Evaluate the config and return a results dict
        """
        # NOTE: checks on config happen before this point
        if num_eval_batches is not None:
            num_eval_batches = int(num_eval_batches)
            if num_eval_batches < 0:
                raise ValueError("num_eval_batches must be None or a nonnegative int")
        if skip_benign:
            raise NotImplementedError
        if skip_attack:
            raise NotImplementedError
        if skip_misclassified:
            raise NotImplementedError

        logger.info("Loading everything for scenario evaluation")

        # Load estimator (optionally with defenses)
        estimator, _ = config_loading.load_model(config["model"])

        defense_config = config.get("defense") or {}
        defense_type = defense_config.get("type")
        if defense_type in ("Preprocessor", "Postprocessor"):
            logger.info(f"Applying internal {defense_type} defense to estimator")
            estimator = config_loading.load_defense_internal(
                config["defense"], estimator
            )

        if config["model"]["fit"]:
            raise NotImplementedError("No model fitting")

        # Load attack (optionally with targeted labels)
        attack_config = config["attack"]
        attack_type = attack_config.get("type")
        targeted = bool(attack_config.get("kwargs", {}).get("targeted"))
        if targeted and attack_config.get("use_label"):
            raise ValueError("Targeted attacks cannot have 'use_label'")
        if attack_type == "preloaded":
            raise NotImplementedError("Ignoring preloaded datasets")
        attack = config_loading.load_attack(attack_config, estimator)
        if targeted != getattr(attack, "targeted", False):
            logger.warning(
                f"targeted config {targeted} != attack field {getattr(attack, 'targeted', False)}"
            )
        if targeted:
            label_targeter = config_loading.load_label_targeter(
                attack_config["targeted_labels"]
            )

        # Load dataset
        logger.info(f"Loading test dataset {config['dataset']['name']}...")
        if config["dataset"]["batch_size"] != 1:
            raise NotImplementedError("batch_size != 1 is not supported.")
        eval_split = config["dataset"].get("eval_split", "test")
        dataset = config_loading.load_dataset(
            config["dataset"],
            epochs=1,
            split=eval_split,
            num_batches=num_eval_batches,
            shuffle_files=False,
        )

        # Load metrics logger
        metrics_logger = metrics.MetricsLogger.from_config(
            config["metric"],  # NOTE: A lot of things happen here!!!
            skip_benign=skip_benign,
            skip_attack=skip_attack,
            targeted=targeted,
        )

        # Load (optional) sample exporter
        # NOTE: probably would work better as a "handler" for the metrics logger
        export_samples = config["scenario"].get("export_samples")
        if export_samples is not None and export_samples > 0:
            sample_exporter = SampleExporter(
                self.scenario_output_dir, dataset.context, export_samples
            )
        else:
            sample_exporter = None

        # Evaluate model on attack and dataset
        logger.info("Running inference on benign and adversarial examples...")
        for x, y in tqdm(dataset, desc="Evaluation"):
            # Benign prediction Stage
            with metrics.resource_context(
                name="Inference",
                profiler=config["metric"].get("profiler_type"),
                computational_resource_dict=metrics_logger.computational_resource_dict,
            ):
                # Ensure that input sample isn't overwritten by estimator
                x.flags.writeable = (
                    False  # probably better to save a defensive copy and then compare
                )
                y_pred = estimator.predict(x)
            metrics_logger.update_task(y, y_pred)  # measure benign task performance

            # Attack stage
            with metrics.resource_context(
                name="Attack",
                profiler=config["metric"].get("profiler_type"),
                computational_resource_dict=metrics_logger.computational_resource_dict,
            ):
                if attack_config.get("use_label"):
                    y_target = y
                elif targeted:
                    y_target = label_targeter.generate(y)
                else:
                    y_target = y_pred  # NOTE: If y is None, then the benign prediction will go again and it will use that prediction as a target
                x_adv = attack.generate(x=x, y=y_target)

            # Adversarial prediction Stage
            # NOTE: we haven't computationally measured here because it should be roughly identical to the benign inference
            # Ensure that input sample isn't overwritten by estimator
            x_adv.flags.writeable = (
                False  # probably better to save a defensive copy and then compare
            )
            y_pred_adv = estimator.predict(x_adv)

            # Measure all of the relevant metrics
            metrics_logger.update_task(
                y, y_pred_adv, adversarial=True
            )  # measure adversarial task performance (e.g., accuracy)
            if targeted:  # measure adversarial targeted task performance
                metrics_logger.update_task(
                    y_target, y_pred_adv, adversarial=True, targeted=True
                )
            metrics_logger.update_perturbation(
                x, x_adv
            )  # measure perturbation (e.g., linf)

            # Export adversarial samples and predictions with their benign counterparts (for comparison)
            if sample_exporter is not None:
                sample_exporter.export(
                    x, x_adv, y, y_pred_adv
                )  # no reason we wouldn't want to also export y_pred or y_target

            # NOTE: This could be an interesting place to stop / go interactive
            #     It could be nice to have a user call next() or repeat()
            #     and interate through the next data batch (or repeat on this batch)

        # Finalize logging results (typically statistics averaged across all samples)
        # NOTE: a lot of gross stuff happens here
        metrics_logger.log_task()  # log benign task
        metrics_logger.log_task(adversarial=True)
        if targeted:
            metrics_logger.log_task(adversarial=True, targeted=True)

        # Output results dictionary as a JSON-able data structure
        # NOTE: additional fun code here
        return metrics_logger.results()
