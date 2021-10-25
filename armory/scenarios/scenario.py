"""
Primary class for scenario
"""

import copy
import json
import logging
import os
import sys
import time
from typing import Optional

from tqdm import tqdm

import armory
from armory import Config, paths
from armory.utils import config_loading, metrics
from armory.utils.export import SampleExporter


logger = logging.getLogger(__name__)


class Scenario:
    """
    Contains the configuration and helper classes needed to execute an Amory evaluation.
    This is the base class of specific tasks like ImageClassificationTask and
    provides significant common processing.
    """

    def __init__(
        self,
        config: Config,
        num_eval_batches: Optional[int] = None,
        skip_benign: Optional[bool] = False,
        skip_attack: Optional[bool] = False,
        skip_misclassified: Optional[bool] = False,
        mongo_host: Optional[str] = None,
        check_run: bool = False,
    ):
        self.check_run = bool(check_run)
        if num_eval_batches is not None and num_eval_batches < 0:
            raise ValueError("num_eval_batches cannot be negative")
        if self.check_run:
            if num_eval_batches:
                raise ValueError("check_run and num_eval_batches are incompatible")
            # Modify dataset entries
            if config["model"]["fit"]:
                config["model"]["fit_kwargs"]["nb_epochs"] = 1
            if config.get("attack", {}).get("type") == "preloaded":
                config["attack"]["check_run"] = True
            # For poisoning scenario
            if config.get("adhoc") and config.get("adhoc").get("train_epochs"):
                config["adhoc"]["train_epochs"] = 1
        self._check_config_and_cli_args(
            config, num_eval_batches, skip_benign, skip_attack, skip_misclassified
        )
        self.config = config
        self._set_output_dir(self.config)
        self.num_eval_batches = num_eval_batches
        self.skip_benign = bool(skip_benign)
        self.skip_attack = bool(skip_attack)
        self.skip_misclassified = bool(skip_misclassified)
        if skip_benign:
            logger.info("Skipping benign classification...")
        if skip_attack:
            logger.info("Skipping attack generation...")
        self.mongo_host = mongo_host
        if self.mongo_host is not None:  # fail fast if pymongo is not installed
            from armory.scenarios import mongo  # noqa: F401

    def _set_output_dir(self, config: Config) -> None:
        runtime_paths = paths.runtime_paths()
        self.scenario_output_dir = os.path.join(
            runtime_paths.output_dir, config["eval_id"]
        )

    def _check_config_and_cli_args(
        self, config, num_eval_batches, skip_benign, skip_attack, skip_misclassified
    ):
        if skip_misclassified:
            if skip_attack or skip_benign:
                raise ValueError(
                    "Cannot pass skip_misclassified if skip_benign or skip_attack is also passed"
                )
            if "categorical_accuracy" not in config["metric"].get("task"):
                raise ValueError(
                    "Cannot pass skip_misclassified if 'categorical_accuracy' metric isn't enabled"
                )
            if config["dataset"].get("batch_size") != 1:
                raise ValueError(
                    "To enable skip_misclassified, 'batch_size' must be set to 1"
                )
            if config["attack"].get("kwargs", {}).get("targeted"):
                raise ValueError("skip_misclassified only works for untargeted attacks")

    def load_model(self, defended=True):
        model_config = self.config["model"]
        model_name = f"{model_config['module']}.{model_config['name']}"
        model, _ = config_loading.load_model(model_config)

        if defended:
            defense_config = self.config.get("defense") or {}
            defense_type = defense_config.get("type")
            if defense_type in ["Preprocessor", "Postprocessor"]:
                logger.info(f"Applying internal {defense_type} defense to model")
                model = config_loading.load_defense_internal(defense_config, model)
            elif defense_type == "Trainer":
                self.trainer = config_loading.load_defense_wrapper(
                    defense_config, model
                )
            elif defense_type is not None:
                raise ValueError(f"{defense_type} not currently supported")
        else:
            logger.info("Not loading any defenses for model")
            defense_type = None

        self.model = model
        self.model_name = model_name
        self.use_fit = bool(model_config["fit"])
        self.fit_kwargs = model_config.get("fit_kwargs", {})
        self.predict_kwargs = model_config.get("predict_kwargs", {})
        self.defense_type = defense_type

    def load_train_dataset(self, train_split_default="train"):
        dataset_config = self.config["dataset"]
        logger.info(f"Loading train dataset {dataset_config['name']}...")
        self.train_dataset = config_loading.load_dataset(
            dataset_config,
            epochs=self.fit_kwargs["nb_epochs"],
            split=dataset_config.get("train_split", train_split_default),
            check_run=self.check_run,
            shuffle_files=True,
        )

    def fit(self):
        if self.defense_type == "Trainer":
            logger.info(f"Training with {type(self.trainer)} Trainer defense...")
            self.trainer.fit_generator(self.train_dataset, **self.fit_kwargs)
        else:
            logger.info(f"Fitting model {self.model_name}...")
            self.model.fit_generator(self.train_dataset, **self.fit_kwargs)

    def load_attack(self):
        attack_config = self.config["attack"]
        attack_type = attack_config.get("type")
        if attack_type == "preloaded" and self.skip_misclassified:
            raise ValueError("Cannot use skip_misclassified with preloaded dataset")

        if attack_type == "preloaded":
            preloaded_split = attack_config.get("kwargs", {}).get(
                "split", "adversarial"
            )
            self.test_dataset = config_loading.load_adversarial_dataset(
                attack_config,
                epochs=1,
                split=preloaded_split,
                num_batches=self.num_eval_batches,
                shuffle_files=False,
            )
            targeted = attack_config.get("targeted", False)
        else:
            attack = config_loading.load_attack(attack_config, self.model)
            self.attack = attack
            targeted = getattr(attack, "targeted", False)
            if targeted:
                label_targeter = config_loading.load_label_targeter(
                    attack_config["targeted_labels"]
                )

        use_label = bool(attack_config.get("use_label"))
        if targeted and use_label:
            raise ValueError("Targeted attacks cannot have 'use_label'")
        generate_kwargs = copy.deepcopy(attack_config.get("generate_kwargs", {}))

        self.attack_type = attack_type
        self.targeted = targeted
        if self.targeted:
            self.label_targeter = label_targeter
        self.use_label = use_label
        self.generate_kwargs = generate_kwargs

    def load_dataset(self, eval_split_default="test"):
        dataset_config = self.config["dataset"]
        eval_split = dataset_config.get("eval_split", eval_split_default)
        # Evaluate the ART model on benign test examples
        logger.info(f"Loading test dataset {dataset_config['name']}...")
        self.test_dataset = config_loading.load_dataset(
            dataset_config,
            epochs=1,
            split=eval_split,
            num_batches=self.num_eval_batches,
            check_run=self.check_run,
            shuffle_files=False,
        )
        self.i = -1

    def load_metrics(self):
        if hasattr(self, "targeted"):
            targeted = self.targeted
        else:
            targeted = False
            logger.warning(
                "Run 'load_attack' before 'load_metrics' if not just doing benign inference"
            )

        metrics_config = self.config["metric"]
        metrics_logger = metrics.MetricsLogger.from_config(
            metrics_config,
            skip_benign=self.skip_benign,
            skip_attack=self.skip_attack,
            targeted=targeted,
        )

        self.profiler_kwargs = dict(
            profiler=metrics_config.get("profiler_type"),
            computational_resource_dict=metrics_logger.computational_resource_dict,
        )

        export_samples = self.config["scenario"].get("export_samples")
        if export_samples is not None and export_samples > 0:
            sample_exporter = SampleExporter(
                self.scenario_output_dir, self.test_dataset.context, export_samples
            )
        else:
            sample_exporter = None

        self.metrics_logger = metrics_logger
        self.sample_exporter = sample_exporter

    def load(self):
        self.load_model()
        if self.use_fit:
            self.load_train_dataset()
            self.fit()
        self.load_attack()
        self.load_dataset()
        self.load_metrics()
        return self

    def evaluate_all(self):
        logger.info("Running inference on benign and adversarial examples")
        for _ in tqdm(range(len(self.test_dataset)), desc="Evaluation"):
            self.next()
            self.evaluate_current()

    def next(self):
        x, y = next(self.test_dataset)
        i = self.i + 1
        self.i, self.x, self.y = i, x, y
        self.y_pred, self.y_target, self.x_adv, self.y_pred_adv = None, None, None, None

    def run_benign(self):
        x, y = self.x, self.y
        x.flags.writeable = False
        with metrics.resource_context(name="Inference", **self.profiler_kwargs):
            y_pred = self.model.predict(x, **self.predict_kwargs)
        self.metrics_logger.update_task(y, y_pred)
        self.y_pred = y_pred

        if self.skip_misclassified:
            self.misclassified = not any(metrics.categorical_accuracy(y, y_pred))

    def run_attack(self):
        x, y, y_pred = self.x, self.y, self.y_pred

        with metrics.resource_context(name="Attack", **self.profiler_kwargs):
            if self.skip_misclassified and self.misclassified:
                y_target = None

                x_adv = x
            elif self.attack_type == "preloaded":
                if self.targeted:
                    y, y_target = y
                else:
                    y_target = None

                if len(x) == 2:
                    x, x_adv = x
                else:
                    x_adv = x
            else:
                if self.use_label:
                    y_target = y
                elif self.targeted:
                    y_target = self.label_targeter.generate(y)
                else:
                    y_target = None

                x_adv = self.attack.generate(x=x, y=y_target, **self.generate_kwargs)

        if self.skip_misclassified and self.misclassified:
            y_pred_adv = y_pred
        else:
            # Ensure that input sample isn't overwritten by model
            x_adv.flags.writeable = False
            y_pred_adv = self.model.predict(x_adv, **self.predict_kwargs)

        self.metrics_logger.update_task(y, y_pred_adv, adversarial=True)
        if self.targeted:
            self.metrics_logger.update_task(
                y_target, y_pred_adv, adversarial=True, targeted=True
            )
        self.metrics_logger.update_perturbation(x, x_adv)

        if self.sample_exporter is not None:
            self.sample_exporter.export(x, x_adv, y, y_pred_adv)

        self.x_adv, self.y_target, self.y_pred_adv = x_adv, y_target, y_pred_adv

    def evaluate_current(self):
        if not self.skip_benign:
            self.run_benign()
        if not self.skip_attack:
            self.run_attack()

    def finalize_results(self):
        metrics_logger = self.metrics_logger
        metrics_logger.log_task()
        metrics_logger.log_task(adversarial=True)
        if self.targeted:
            metrics_logger.log_task(adversarial=True, targeted=True)
        self.results = metrics_logger.results()

    def _evaluate(self) -> dict:
        """
        Evaluate the config and return a results dict
        """
        self.load()
        self.evaluate_all()
        self.finalize_results()
        return self.results

    def evaluate(self):
        """
        Evaluate a config for robustness against attack.
        """
        try:
            results = self._evaluate()
        except Exception as e:
            if str(e) == "assignment destination is read-only":
                logger.exception(
                    "Encountered error during scenario evaluation. Be sure "
                    + "that the classifier's predict() isn't directly modifying the "
                    + "input variable itself, as this can cause unexpected behavior in ART."
                )
            else:
                logger.exception("Encountered error during scenario evaluation.")
            sys.exit(1)

        if results is None:
            logger.warning(f"{self._evaluate} returned None, not a dict")
        output = self._prepare_results(self.config, results)
        self._save(output)
        if self.mongo_host is not None:
            self._send_to_mongo(self.mongo_host, output)

    def _prepare_results(self, config: dict, results: dict, adv_examples=None) -> dict:
        """
        Build the JSON results blob for _save() and _send_to_mongo()

        adv_examples are (optional) instances of the actual examples used.
            They will be saved in a binary format.
        """
        if adv_examples is not None:
            raise NotImplementedError("saving adversarial examples")

        timestamp = int(time.time())
        output = {
            "armory_version": armory.__version__,
            "config": config,
            "results": results,
            "timestamp": timestamp,
        }
        return output

    def _save(self, output: dict):
        """
        Save json-formattable output to a file
        """
        override_name = output["config"]["sysconfig"].get("output_filename", None)
        scenario_name = (
            override_name if override_name else output["config"]["scenario"]["name"]
        )
        filename = f"{scenario_name}_{output['timestamp']}.json"
        logger.info(
            "Saving evaluation results to path\n"
            f"{self.scenario_output_dir}/{filename}\n"
            "inside container."
        )
        with open(os.path.join(self.scenario_output_dir, filename), "w") as f:
            f.write(json.dumps(output, sort_keys=True, indent=4) + "\n")

    def _send_to_mongo(self, output: dict):
        """
        Send results to a Mongo database at mongo_host
        """
        import mongo

        mongo.send_to_db(output, self.mongo_host)
