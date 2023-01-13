"""
Primary class for scenario
"""

import copy
import importlib
import os
import sys
import time
from typing import Optional

from tqdm import tqdm

import armory
from armory import Config, metrics, paths
from armory.instrument import MetricsLogger, del_globals, get_hub, get_probe
from armory.instrument.export import ExportMeter, PredictionMeter
from armory.logs import log
from armory.metrics import compute
from armory.utils import config_loading, json_utils


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
        check_run: bool = False,
    ):
        self.probe = get_probe("scenario")
        self.hub = get_hub()
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

            for attack_kwarg in [
                "max_iter",
                "max_iter_1",
                "max_iter_2",
                "max_epochs",
                "max_trials",
                "model_retraining_epoch",
            ]:
                if config.get("attack", {}).get("kwargs", {}).get(attack_kwarg):
                    config["attack"]["kwargs"][attack_kwarg] = 1

            # For poisoning scenario
            if config.get("adhoc") and config.get("adhoc").get("train_epochs"):
                config["adhoc"]["train_epochs"] = 1
        self._check_config_and_cli_args(
            config, num_eval_batches, skip_benign, skip_attack, skip_misclassified
        )
        self.config = config
        self.num_eval_batches = num_eval_batches
        self.skip_benign = bool(skip_benign)
        self.skip_attack = bool(skip_attack)
        self.skip_misclassified = bool(skip_misclassified)
        if skip_benign:
            log.info("Skipping benign classification...")
        if skip_attack:
            log.info("Skipping attack generation...")
        self.time_stamp = time.time()
        self.export_subdir = "saved_samples"
        self._set_output_dir(self.config.get("eval_id"))
        if os.path.exists(f"{self.scenario_output_dir}/{self.export_subdir}"):
            log.warning(
                f"Export output directory {self.scenario_output_dir}/{self.export_subdir} already exists, will create new directory"
            )
            self._set_export_dir(f"{self.export_subdir}_{self.time_stamp}")
        self.results = None

    def user_init(self) -> None:
        """
        Import the user-specified initialization module
            and (optionally) call the specified function name with kwargs
        """
        user_init = self.config.get("user_init")
        if user_init is not None:
            module = user_init.get("module")
            log.info(f"Importing user_init module {module}")
            if not isinstance(module, str):
                raise ValueError("config: 'user_init' field 'module' must be a str")
            try:
                mod = importlib.import_module(module)
            except ModuleNotFoundError as err:
                raise ValueError(
                    f"config: 'user_init' field 'module' '{module}' cannot be imported."
                    " If using docker, does it need to be added to"
                    " config['sysconfig']['external_github_repo']?"
                ) from err
            name = user_init.get("name")
            kwargs = user_init.get("kwargs") or {}
            if name:
                if kwargs:
                    kwargs_str = f"**{kwargs}"
                else:
                    kwargs_str = ""
                log.info(f"Calling user_init function {module}.{name}({kwargs_str})")
                target = getattr(mod, name, None)
                if target is None:
                    raise ValueError(f"user_init name {name} cannot be found")
                if not callable(target):
                    raise ValueError(f"{module}.{name} is not callable")
                target(**kwargs)
            elif kwargs:
                log.warning("Ignoring user_init kwargs because name is False")

    def _set_output_dir(self, eval_id) -> None:
        runtime_paths = paths.runtime_paths()
        self.scenario_output_dir = os.path.join(runtime_paths.output_dir, eval_id)
        self.hub._set_output_dir(self.scenario_output_dir)
        self._set_export_dir(self.export_subdir)

    def _set_export_dir(self, output_subdir) -> None:
        self.export_dir = f"{self.scenario_output_dir}/{output_subdir}"
        self.export_subdir = output_subdir
        self.hub._set_export_dir(output_subdir)

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
                log.info(f"Applying internal {defense_type} defense to model")
                model = config_loading.load_defense_internal(defense_config, model)
            elif defense_type == "Trainer":
                self.trainer = config_loading.load_defense_wrapper(
                    defense_config, model
                )
            elif defense_type is not None:
                raise ValueError(f"{defense_type} not currently supported")
        else:
            log.info("Not loading any defenses for model")
            defense_type = None

        self.model = model
        self.model_name = model_name
        self.use_fit = bool(model_config["fit"])
        self.fit_kwargs = model_config.get("fit_kwargs", {})
        self.predict_kwargs = model_config.get("predict_kwargs", {})
        self.defense_type = defense_type

    def load_train_dataset(self, train_split_default="train"):
        dataset_config = self.config["dataset"]
        log.info(f"Loading train dataset {dataset_config['name']}...")
        self.train_dataset = config_loading.load_dataset(
            dataset_config,
            epochs=self.fit_kwargs["nb_epochs"],
            split=dataset_config.get("train_split", train_split_default),
            check_run=self.check_run,
            shuffle_files=True,
        )

    def fit(self):
        if self.defense_type == "Trainer":
            log.info(f"Training with {type(self.trainer)} Trainer defense...")
            self.trainer.fit_generator(self.train_dataset, **self.fit_kwargs)
        else:
            log.info(f"Fitting model {self.model_name}...")
            self.model.fit_generator(self.train_dataset, **self.fit_kwargs)

    def load_attack(self):
        attack_config = self.config["attack"]
        attack_type = attack_config.get("type")
        if attack_type == "preloaded" and self.skip_misclassified:
            raise ValueError("Cannot use skip_misclassified with preloaded dataset")

        if "summary_writer" in attack_config.get("kwargs", {}):
            summary_writer_kwarg = attack_config.get("kwargs").get("summary_writer")
            if isinstance(summary_writer_kwarg, str):
                log.warning(
                    f"Overriding 'summary_writer' attack kwarg {summary_writer_kwarg} with {self.scenario_output_dir}."
                )
            attack_config["kwargs"][
                "summary_writer"
            ] = f"{self.scenario_output_dir}/tfevents_{self.time_stamp}"
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
            targeted = attack_config.get("kwargs", {}).get("targeted", False)
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
        log.info(f"Loading test dataset {dataset_config['name']}...")
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
        if not hasattr(self, "targeted"):
            log.warning(
                "Run 'load_attack' before 'load_metrics' if not just doing benign inference"
            )

        metrics_config = self.config["metric"]
        metrics_logger = MetricsLogger.from_config(
            metrics_config,
            include_benign=not self.skip_benign,
            include_adversarial=not self.skip_attack,
            include_targeted=self.targeted,
        )
        self.profiler = compute.profiler_from_config(metrics_config)
        self.metrics_logger = metrics_logger

    def load_export_meters(self):
        if self.config["scenario"].get("export_samples") is not None:
            log.warning(
                "The export_samples field was deprecated in Armory 0.15.0. Please use export_batches instead."
            )

        num_export_batches = self.config["scenario"].get("export_batches", 0)
        if num_export_batches is True:
            num_export_batches = len(self.test_dataset)
        self.num_export_batches = int(num_export_batches)
        self.sample_exporter = self._load_sample_exporter()

        for probe_value in ["x", "x_adv"]:
            export_meter = ExportMeter(
                f"{probe_value}_exporter",
                self.sample_exporter,
                f"scenario.{probe_value}",
                max_batches=self.num_export_batches,
            )
            self.hub.connect_meter(export_meter, use_default_writers=False)
            if self.skip_attack:
                break

        pred_meter = PredictionMeter(
            "pred_dict_exporter",
            self.export_dir,
            y_probe="scenario.y",
            y_pred_clean_probe="scenario.y_pred" if not self.skip_benign else None,
            y_pred_adv_probe="scenario.y_pred_adv" if not self.skip_attack else None,
            max_batches=self.num_export_batches,
        )
        self.hub.connect_meter(pred_meter, use_default_writers=False)

    def _load_sample_exporter(self):
        raise NotImplementedError(
            f"_load_sample_exporter() method is not implemented for scenario {self.__class__}"
        )

    def load(self):
        self.user_init()
        self.load_model()
        if self.use_fit:
            self.load_train_dataset()
            self.fit()
        self.load_attack()
        self.load_dataset()
        self.load_metrics()
        self.load_export_meters()
        return self

    def evaluate_all(self):
        log.info("Running inference on benign and adversarial examples")
        for _ in tqdm(range(len(self.test_dataset)), desc="Evaluation"):
            self.next()
            self.evaluate_current()
        self.hub.set_context(stage="finished")

    def next(self):
        self.hub.set_context(stage="next")
        x, y = next(self.test_dataset)
        i = self.i + 1
        self.hub.set_context(batch=i)
        self.i, self.x, self.y = i, x, y
        self.probe.update(i=i, x=x, y=y)
        self.y_pred, self.y_target, self.x_adv, self.y_pred_adv = None, None, None, None

    def _check_x(self, function_name):
        if not hasattr(self, "x"):
            raise ValueError(f"Run `next()` before `{function_name}()`")

    def run_benign(self):
        self._check_x("run_benign")
        self.hub.set_context(stage="benign")

        x, y = self.x, self.y
        x.flags.writeable = False
        with self.profiler.measure("Inference"):
            y_pred = self.model.predict(x, **self.predict_kwargs)
        self.y_pred = y_pred
        self.probe.update(y_pred=y_pred)

        if self.skip_misclassified:
            self.misclassified = not any(
                metrics.task.batch.categorical_accuracy(y, y_pred)
            )

    def run_attack(self):
        self._check_x("run_attack")
        self.hub.set_context(stage="attack")
        x, y, y_pred = self.x, self.y, self.y_pred

        with self.profiler.measure("Attack"):
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

        self.hub.set_context(stage="adversarial")
        if self.skip_misclassified and self.misclassified:
            y_pred_adv = y_pred
        else:
            # Ensure that input sample isn't overwritten by model
            x_adv.flags.writeable = False
            y_pred_adv = self.model.predict(x_adv, **self.predict_kwargs)

        self.probe.update(x_adv=x_adv, y_pred_adv=y_pred_adv)
        if self.targeted:
            self.probe.update(y_target=y_target)

        self.x_adv, self.y_target, self.y_pred_adv = x_adv, y_target, y_pred_adv

    def evaluate_current(self):
        self._check_x("evaluate_current")
        if not self.skip_benign:
            self.run_benign()
        if not self.skip_attack:
            self.run_attack()

    def finalize_results(self):
        self.metric_results = self.metrics_logger.results()
        self.compute_results = self.profiler.results()
        self.results = {}
        self.results.update(self.metric_results)
        self.results["compute"] = self.compute_results

    def _evaluate(self) -> dict:
        """
        Evaluate the config and set the results dict self.results
        """
        self.load()
        self.evaluate_all()
        self.finalize_results()
        log.debug("Clearing global instrumentation variables")
        del_globals()

    def evaluate(self):
        """
        Evaluate a config for robustness against attack and save results JSON
        """
        try:
            self._evaluate()
        except Exception as e:
            if str(e) == "assignment destination is read-only":
                log.exception(
                    "Encountered error during scenario evaluation. Be sure "
                    + "that the classifier's predict() isn't directly modifying the "
                    + "input variable itself, as this can cause unexpected behavior in ART."
                )
            else:
                log.exception("Encountered error during scenario evaluation.")
            sys.exit(1)

        if self.results is None:
            log.warning(f"{self._evaluate} did not set self.results to a dict")

        self.save()

    def prepare_results(self) -> dict:
        """
        Return the JSON results blob to be used in save() method
        """
        if not hasattr(self, "results"):
            raise AttributeError(
                "Results have not been finalized. Please call "
                "finalize_results() before saving output."
            )

        output = {
            "armory_version": armory.__version__,
            "config": self.config,
            "results": self.results,
            "timestamp": int(self.time_stamp),
        }
        return output

    def save(self):
        """
        Write results JSON file to Armory scenario output directory
        """
        output = self.prepare_results()

        override_name = output["config"]["sysconfig"].get("output_filename", None)
        scenario_name = (
            override_name if override_name else output["config"]["scenario"]["name"]
        )
        filename = f"{scenario_name}_{output['timestamp']}.json"
        log.info(
            "Saving evaluation results to path "
            f"{self.scenario_output_dir}/{filename} "
            "inside container."
        )
        output_path = os.path.join(self.scenario_output_dir, filename)
        with open(output_path, "w") as f:
            json_utils.dump(output, f)
        if os.path.getsize(output_path) > 2**27:
            log.warning(
                "Results json file exceeds 128 MB! "
                "Recommend checking what is being recorded!"
            )
