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
    load_defense_wrapper,
    load_defense_internal,
    load_label_targeter,
)
from armory.utils import metrics
from armory.scenarios.base import Scenario
from armory.utils.export import SampleExporter

logger = logging.getLogger(__name__)


class AutomaticSpeechRecognition(Scenario):
    def _evaluate(
        self,
        config: dict,
        num_eval_batches: Optional[int],
        skip_benign: Optional[bool],
        skip_attack: Optional[bool],
    ) -> dict:
        """
        Evaluate the config and return a results dict
        """
        model_config = config["model"]
        estimator, fit_preprocessing_fn = load_model(model_config)

        defense_config = config.get("defense") or {}
        defense_type = defense_config.get("type")

        if defense_type in ["Preprocessor", "Postprocessor"]:
            logger.info(f"Applying internal {defense_type} defense to estimator")
            estimator = load_defense_internal(config["defense"], estimator)

        if model_config["fit"]:
            try:
                estimator.set_learning_phase(True)
            except NotImplementedError:
                logger.exception("set_learning_phase error; training may not work.")

            logger.info(
                f"Fitting model {model_config['module']}.{model_config['name']}..."
            )
            fit_kwargs = model_config["fit_kwargs"]

            logger.info(f"Loading train dataset {config['dataset']['name']}...")
            batch_size = config["dataset"].pop("batch_size")
            config["dataset"]["batch_size"] = fit_kwargs.get(
                "fit_batch_size", batch_size
            )
            train_data = load_dataset(
                config["dataset"],
                epochs=fit_kwargs["nb_epochs"],
                split=config["dataset"].get("train_split", "train_clean100"),
                preprocessing_fn=fit_preprocessing_fn,
                shuffle_files=True,
            )
            config["dataset"]["batch_size"] = batch_size
            if defense_type == "Trainer":
                logger.info(f"Training with {defense_type} defense...")
                defense = load_defense_wrapper(config["defense"], estimator)
                defense.fit_generator(train_data, **fit_kwargs)
            else:
                logger.info("Fitting estimator on clean train dataset...")
                estimator.fit_generator(train_data, **fit_kwargs)

        if defense_type == "Transform":
            # NOTE: Transform currently not supported
            logger.info(f"Transforming estimator with {defense_type} defense...")
            defense = load_defense_wrapper(config["defense"], estimator)
            estimator = defense()

        try:
            estimator.set_learning_phase(False)
        except NotImplementedError:
            logger.warning(
                "Unable to set estimator's learning phase. As of ART 1.4.1, "
                "this is not yet supported for speech recognition models."
            )

        attack_config = config["attack"]
        attack_type = attack_config.get("type")

        targeted = bool(attack_config.get("targeted"))
        metrics_logger = metrics.MetricsLogger.from_config(
            config["metric"],
            skip_benign=skip_benign,
            skip_attack=skip_attack,
            targeted=targeted,
        )

        if config["dataset"]["batch_size"] != 1:
            logger.warning("Evaluation batch_size != 1 may not be supported.")

        predict_kwargs = config["model"].get("predict_kwargs", {})
        eval_split = config["dataset"].get("eval_split", "test_clean")
        if skip_benign:
            logger.info("Skipping benign classification...")
        else:
            # Evaluate the ART estimator on benign test examples
            logger.info(f"Loading test dataset {config['dataset']['name']}...")
            test_data = load_dataset(
                config["dataset"],
                epochs=1,
                split=eval_split,
                num_batches=num_eval_batches,
                shuffle_files=False,
            )
            logger.info("Running inference on benign examples...")
            for x, y in tqdm(test_data, desc="Benign"):
                # Ensure that input sample isn't overwritten by estimator
                x.flags.writeable = False
                with metrics.resource_context(
                    name="Inference",
                    profiler=config["metric"].get("profiler_type"),
                    computational_resource_dict=metrics_logger.computational_resource_dict,
                ):
                    y_pred = estimator.predict(x, **predict_kwargs)
                metrics_logger.update_task(y, y_pred)
            metrics_logger.log_task()

        if skip_attack:
            logger.info("Skipping attack generation...")
            return metrics_logger.results()

        # Imperceptible attack still WIP
        if (config.get("adhoc") or {}).get("skip_adversarial"):
            logger.info("Skipping adversarial classification...")
            return metrics_logger.results()

        # Evaluate the ART estimator on adversarial test examples
        logger.info("Generating or loading / testing adversarial examples...")

        if attack_type == "preloaded":
            test_data = load_adversarial_dataset(
                attack_config,
                epochs=1,
                split="adversarial",
                num_batches=num_eval_batches,
                shuffle_files=False,
            )
        else:
            attack = load_attack(attack_config, estimator)
            if targeted != attack.targeted:
                logger.warning(
                    f"targeted config {targeted} != attack field {attack.targeted}"
                )
            test_data = load_dataset(
                config["dataset"],
                epochs=1,
                split=eval_split,
                num_batches=num_eval_batches,
                shuffle_files=False,
            )
            if targeted:
                label_targeter = load_label_targeter(attack_config["targeted_labels"])

        export_samples = config["scenario"].get("export_samples")
        if export_samples is not None and export_samples > 0:
            sample_exporter = SampleExporter(
                self.scenario_output_dir, test_data.context, export_samples
            )
        else:
            sample_exporter = None

        for x, y in tqdm(test_data, desc="Attack"):
            with metrics.resource_context(
                name="Attack",
                profiler=config["metric"].get("profiler_type"),
                computational_resource_dict=metrics_logger.computational_resource_dict,
            ):
                if attack_type == "preloaded":
                    x, x_adv = x
                    if targeted:
                        y, y_target = y
                elif attack_config.get("use_label"):
                    x_adv = attack.generate(x=x, y=y)
                elif targeted:
                    y_target = label_targeter.generate(y)
                    x_adv = attack.generate(x=x, y=y_target)
                else:
                    x_adv = attack.generate(x=x)

            # Ensure that input sample isn't overwritten by estimator
            x_adv.flags.writeable = False
            y_pred_adv = estimator.predict(x_adv, **predict_kwargs)
            metrics_logger.update_task(y, y_pred_adv, adversarial=True)
            if targeted:
                metrics_logger.update_task(
                    y_target, y_pred_adv, adversarial=True, targeted=True,
                )
            metrics_logger.update_perturbation(x, x_adv)
            if sample_exporter is not None:
                sample_exporter.export(x, x_adv, y, y_pred_adv)
        metrics_logger.log_task(adversarial=True)
        if targeted:
            metrics_logger.log_task(adversarial=True, targeted=True)
        return metrics_logger.results()
