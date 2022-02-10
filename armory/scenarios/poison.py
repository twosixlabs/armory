"""
Extended scenario for poisoning
"""

import copy
import logging
import os
import random
from typing import Optional


import numpy as np
import torch


import armory.utils.poisoning as poisoning_utils
from armory.scenarios.scenario import Scenario
from armory.scenarios.utils import to_categorical
from armory.utils import config_loading, metrics
from armory.utils.export import SampleExporter


logger = logging.getLogger(__name__)


class DatasetPoisoner:
    def __init__(self, attack, source_class, target_class, fraction=1.0):
        """
        fraction inputs of source class have a trigger added and are labeled as target
        """
        self.attack = attack
        self.source_class = source_class
        self.target_class = target_class
        self.fraction = self._check_fraction(fraction)

    def _check_fraction(self, fraction):
        fraction = float(fraction)
        if not (0.0 <= fraction <= 1.0):
            raise ValueError(f"fraction {fraction} not in [0.0, 1.0] range")
        return fraction

    def get_poison_index(self, y, fraction=None):
        if fraction is None:
            fraction = self.fraction
        else:
            fraction = self._check_fraction(fraction)

        source_index = np.where(y == self.source_class)[0]
        total = len(source_index)
        poison_count = int(fraction * total)
        if poison_count == 0:
            logger.warning(f"0 of {total} poisoned for class {self.source_class}.")
        return np.sort(np.random.choice(source_index, size=poison_count, replace=False))

    def poison_dataset(self, x, y, return_index=False, fraction=None):
        """
        Return a poisoned version of dataset x, y
            if return_index, return x, y, index
        If fraction is not None, use it to override default
        """
        if len(x) != len(y):
            raise ValueError("Sizes of x and y do not match")
        poison_x, poison_y = list(x), list(y)
        poison_index = self.get_poison_index(y, fraction=fraction)
        for i in poison_index:
            poison_x_i, poison_y[i] = self.attack.poison(x[i], [self.target_class])
            poison_x[i] = np.asarray(poison_x_i, dtype=x[i].dtype)
        poison_x, poison_y = np.array(poison_x), np.array(poison_y, dtype=int)

        if return_index:
            return poison_x, poison_y, poison_index
        return poison_x, poison_y


class Poison(Scenario):
    def __init__(
        self,
        config: dict,
        num_eval_batches: Optional[int] = None,
        skip_benign: Optional[bool] = False,
        skip_attack: Optional[bool] = False,
        skip_misclassified: Optional[bool] = False,
        triggered: Optional[bool] = True,
        **kwargs,
    ):
        if num_eval_batches:
            raise ValueError("num_eval_batches shouldn't be set for poisoning scenario")
        if skip_benign:
            raise ValueError("skip_benign shouldn't be set for poisoning scenario")
        if skip_attack:
            raise ValueError("skip_attack shouldn't be set for poisoning scenario")
        if skip_misclassified:
            raise ValueError(
                "skip_misclassified shouldn't be set for poisoning scenario"
            )
        super().__init__(config, **kwargs)
        if triggered is None:
            triggered = True
        else:
            triggered = bool(triggered)
        if not triggered:
            raise NotImplementedError("triggered=False attacks are not implemented")
        self.triggered = triggered

    def set_random_seed(self):
        # Set random seed due to large variance in attack and defense success
        self.seed = self.config["adhoc"]["split_id"]
        np.random.seed(self.seed)
        random.seed(self.seed)
        if self.config["sysconfig"].get("use_gpu"):
            os.environ["TF_CUDNN_DETERMINISM"] = "1"

    def load_model(self, defended=True):
        # Scenario assumes canonical preprocessing_fn is used makes images all same size
        model_config = self.config["model"]
        model, _ = config_loading.load_model(model_config)

        if defended:
            defense_config = self.config.get("defense") or {}
            if "data_augmentation" in defense_config:
                for data_aug_config in defense_config["data_augmentation"].values():
                    model = config_loading.load_defense_internal(data_aug_config, model)
                logger.info(
                    f"model.preprocessing_defences: {model.preprocessing_defences}"
                )
        self.model = model
        self.predict_kwargs = model_config.get("predict_kwargs", {})

    def set_dataset_kwargs(self):
        self.dataset_kwargs = dict(epochs=1, shuffle_files=False,)

    def load_train_dataset(self, train_split_default=None):
        """
        Load and create in memory dataset
            detect_poison does not currently support data generators
        """
        if train_split_default is not None:
            raise ValueError(
                "train_split_default not used in this loading method for poison"
            )
        adhoc_config = self.config.get("adhoc") or {}
        self.train_epochs = adhoc_config["train_epochs"]
        self.fit_batch_size = adhoc_config.get(
            "fit_batch_size", self.config["dataset"]["batch_size"]
        )

        # Flag to determine whether training uses categorical or sparse labels
        self.categorical_labels = adhoc_config.get("defense_categorical_labels", True)
        if self.categorical_labels:
            self.label_function = to_categorical
        else:
            self.label_function = lambda y: y

        dataset_config = self.config["dataset"]
        logger.info(f"Loading dataset {dataset_config['name']}...")
        ds = config_loading.load_dataset(
            dataset_config,
            split=dataset_config.get("train_split", "train"),
            **self.dataset_kwargs,
        )
        self.x_clean, self.y_clean = (np.concatenate(z, axis=0) for z in zip(*list(ds)))

    def load_poisoner(self):
        adhoc_config = self.config.get("adhoc") or {}
        attack_config = self.config["attack"]
        if attack_config.get("type") == "preloaded":
            raise ValueError("preloaded attacks not currently supported for poisoning")

        self.use_poison = bool(adhoc_config["poison_dataset"])
        self.source_class = adhoc_config["source_class"]
        self.target_class = adhoc_config["target_class"]
        if self.use_poison:
            attack = config_loading.load(attack_config)
            self.poisoner = DatasetPoisoner(
                attack,
                self.source_class,
                self.target_class,
                fraction=adhoc_config["fraction_poisoned"],
            )
            self.test_poisoner = self.poisoner

    def poison_dataset(self):
        if self.use_poison:
            (
                self.x_poison,
                self.y_poison,
                self.poison_index,
            ) = self.poisoner.poison_dataset(
                self.x_clean, self.y_clean, return_index=True
            )
        else:
            self.x_poison, self.y_poison, self.poison_index = (
                self.x_clean,
                self.y_clean,
                np.array([]),
            )

    def filter_dataset(self):
        adhoc_config = self.config["adhoc"]
        # filtering defense requires more than a single batch to run properly
        if (
            adhoc_config.get("use_poison_filtering_defense", True) and not self.check_run
        ):
            defense_config = copy.deepcopy(self.config["defense"] or {})
            if "data_augmentation" in defense_config:
                defense_config.pop("data_augmentation")  # NOTE: RESISC10 ONLY

            # Assumes classifier_for_defense and classifier use same preprocessing function
            defense_model_config = adhoc_config.get(
                "defense_model", self.config["model"]
            )
            classifier_for_defense, _ = config_loading.load_model(defense_model_config)
            logger.info(
                f"Fitting model {defense_model_config['module']}.{defense_model_config['name']} "
                f"for defense {defense_config['name']}..."
            )
            # Flag to determine whether defense_classifier is trained directly
            #     (default API) or is trained as part of detect_poisons method
            if adhoc_config.get("fit_defense_classifier_outside_defense", True):
                classifier_for_defense.fit(
                    self.x_poison,
                    self.label_function(self.y_poison),
                    batch_size=self.fit_batch_size,
                    nb_epochs=adhoc_config.get(
                        "defense_train_epochs", self.train_epochs
                    ),
                    verbose=False,
                    shuffle=True,
                )
            defense_fn = config_loading.load_fn(defense_config)
            defense = defense_fn(
                classifier_for_defense,
                self.x_poison,
                self.label_function(self.y_poison),
            )

            detection_kwargs = adhoc_config.get("detection_kwargs", {})
            _, is_clean = defense.detect_poison(**detection_kwargs)
            is_clean = np.array(is_clean)
            logger.info(f"Total clean data points: {np.sum(is_clean)}")
            is_dirty = is_clean.astype(np.int64) == 0
            logger.info(f"Total dirty data points: {np.sum(is_dirty)}")

            logger.info("Filtering out detected poisoned samples")
            indices_to_keep = (is_clean == 1)

            self.filter_perplexity.add_results(self.y_clean, self.poison_index, is_dirty)

        else:
            logger.info(
                "Defense does not require filtering. Model fitting will use all data."
            )
            indices_to_keep = np.ones_like(self.y_poison, dtype=np.bool_)

        # TODO: measure TP and FP rates for filtering
        self.x_train = self.x_poison[indices_to_keep]
        self.y_train = self.y_poison[indices_to_keep]
        self.indices_to_keep = indices_to_keep

    def fit(self):
        if len(self.x_train):
            logger.info("Fitting model")
            self.model.fit(
                self.x_train,
                self.label_function(self.y_train),
                batch_size=self.fit_batch_size,
                nb_epochs=self.train_epochs,
                verbose=False,
                shuffle=True,
            )
        else:
            logger.warning("All data points filtered by defense. Skipping training")

    def load_attack(self):
        raise NotImplementedError(
            "Not implemented for poisoning scenario. Use load_poisoner"
        )

    def load_dataset(self, eval_split_default="test"):
        dataset_config = self.config["dataset"]
        eval_split = dataset_config.get("eval_split", eval_split_default)
        # Evaluate the ART model on benign test examples
        logger.info(f"Loading test dataset {dataset_config['name']}...")
        self.test_dataset = config_loading.load_dataset(
            dataset_config,
            split=eval_split,
            num_batches=self.num_eval_batches,
            **self.dataset_kwargs,
        )
        self.i = -1

        export_samples = self.config["scenario"].get("export_samples")
        if export_samples is not None and export_samples > 0:
            sample_exporter = SampleExporter(
                self.scenario_output_dir, self.test_dataset.context, export_samples
            )
        else:
            sample_exporter = None
        self.sample_exporter = sample_exporter


    def load_metrics(self):
        self.benign_validation_metric = metrics.MetricList("categorical_accuracy")
        self.target_class_benign_metric = metrics.MetricList("categorical_accuracy")
        if self.use_poison:
            self.poisoned_test_metric = metrics.MetricList("categorical_accuracy")
            self.poisoned_targeted_test_metric = metrics.MetricList(
                "categorical_accuracy"
            )
        if self.config["adhoc"].get("use_poison_filtering_defense", False):
            self.filter_perplexity = metrics.MetricList("filter_perplexity_fps_benign")
        explanatory_model_config = self.config.get("explanatory_model", False)
        if explanatory_model_config:
            explanatory_model, _ = poisoning_utils.load_explanatory_model(explanatory_model_config)
            self.explanatory_model = explanatory_model
            self.explanatory_model_resize_image = explanatory_model_config.get("resize_image", True)
            self.majority_x_class_prediction_chi2_metrics = {
                class_id: metrics.MetricList("poison_chi2_p_value") for class_id in np.unique(self.y_clean)
            }
            self.majority_x_class_prediction_spd_metrics = {
                class_id: metrics.MetricList("poison_spd") for class_id in np.unique(self.y_clean)
            }
            if self.config["adhoc"].get("use_poison_filtering_defense", False):
                self.majority_x_passed_filter_chi2_metrics = {
                    class_id: metrics.MetricList("poison_chi2_p_value") for class_id in np.unique(self.y_clean)
                }
                self.majority_x_passed_filter_spd_metrics = {
                    class_id: metrics.MetricList("poison_spd") for class_id in np.unique(self.y_clean)
                }

    def load(self):
        self.set_random_seed()
        self.set_dataset_kwargs()
        self.load_model()
        self.load_train_dataset()
        self.load_poisoner()
        self.load_metrics()
        self.poison_dataset()
        self.filter_dataset()
        self.fit()
        self.load_dataset()

    def run_benign(self):
        x, y = self.x, self.y

        x.flags.writeable = False
        y_pred = self.model.predict(x, **self.predict_kwargs)

        self.benign_validation_metric.add_results(y, y_pred)
        source = y == self.source_class
        # NOTE: uses source->target trigger
        if source.any():
            self.target_class_benign_metric.add_results(y[source], y_pred[source])

        self.y_pred = y_pred
        self.source = source

    def run_attack(self):
        x, y = self.x, self.y
        source = self.source

        x_adv, _ = self.test_poisoner.poison_dataset(x, y, fraction=1.0)
        x_adv.flags.writeable = False
        y_pred_adv = self.model.predict(x_adv, **self.predict_kwargs)

        self.poisoned_test_metric.add_results(y, y_pred_adv)
        # NOTE: uses source->target trigger
        if source.any():
            self.poisoned_targeted_test_metric.add_results(
                [self.target_class] * source.sum(), y_pred_adv[source]
            )

        # Note: Will still output x_adv even if it is the same as x, i.e. not poisoned
        if self.sample_exporter is not None:
            self.sample_exporter.export(x, x_adv, y, y_pred_adv)

        self.x_adv = x_adv
        self.y_pred_adv = y_pred_adv

    def evaluate_current(self):
        self.run_benign()
        if self.use_poison:
            self.run_attack()

    def finalize_results(self):
        logger.info(
            f"Unpoisoned validation accuracy: {self.benign_validation_metric.mean():.2%}"
        )
        logger.info(
            f"Unpoisoned validation accuracy on targeted class: {self.target_class_benign_metric.mean():.2%}"
        )
        results = {
            "benign_validation_accuracy": self.benign_validation_metric.mean(),
            "benign_validation_accuracy_targeted_class": self.target_class_benign_metric.mean(),
        }
        if self.use_poison:
            results["poisoned_test_accuracy"] = self.poisoned_test_metric.mean()
            results[
                "poisoned_targeted_misclassification_accuracy"
            ] = self.poisoned_targeted_test_metric.mean()
            logger.info(f"Test accuracy: {self.poisoned_test_metric.mean():.2%}")
            logger.info(
                f"Test targeted misclassification accuracy: {self.poisoned_targeted_test_metric.mean():.2%}"
            )
        if hasattr(self, "filter_perplexity") and not self.check_run:
            results["filter_perplexity"] = self.filter_perplexity.mean()
            logger.info(
                f"Normalized filter perplexity: {self.filter_perplexity.mean()}"
            )

        if hasattr(self, "explanatory_model") and hasattr(self, "poison_index"):
            DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

            # Get majority_ceilings from train set now,
            # because we also need them for the test-set metric
            poisoned_mask = np.zeros_like(self.y_poison, dtype=np.bool_)
            poisoned_mask[self.poison_index.astype(np.int64)] = True
            x_unpoisoned = self.x_poison[~poisoned_mask]
            y_unpoisoned = self.y_poison[~poisoned_mask]
            majority_mask_unpoisoned, majority_ceilings = poisoning_utils.get_majority_mask(
                    explanatory_model=self.explanatory_model,
                    data=list(zip(x_unpoisoned, y_unpoisoned)),
                    device=DEVICE,
                    resize_image=self.explanatory_model_resize_image
                    )
                        
            # model predictions metric
            # Uses predictions and majority labels on TEST set
            # First, reload the test dataset
            dataset_config = self.config["dataset"]
            eval_split = dataset_config.get("eval_split", "test")
            test_dataset = config_loading.load_dataset(
                dataset_config,
                split=eval_split,
                num_batches=self.num_eval_batches,
                **self.dataset_kwargs,
            )
            test_x, test_y = (np.concatenate(z, axis=0) for z in zip(*list(test_dataset)))
            test_set_preds = self.model.predict(test_x, **self.predict_kwargs).argmax(1)
            correct_prediction_mask_test_set = (test_y == test_set_preds)
            majority_mask_test_set, majority_ceilings = poisoning_utils.get_majority_mask(
                explanatory_model=self.explanatory_model,
                data=list(zip(test_x, test_y)),
                majority_ceilings=majority_ceilings, # use ceilings computed from train set
                device=DEVICE,
                resize_image=self.explanatory_model_resize_image
                )

            majority_x_correct_prediction_tables = metrics.make_contingency_tables(
                test_y,
                majority_mask_test_set,
                correct_prediction_mask_test_set
            )

            for class_id in np.unique(test_y):
                self.majority_x_class_prediction_chi2_metrics[class_id].add_results(majority_x_correct_prediction_tables[class_id])
                self.majority_x_class_prediction_spd_metrics[class_id].add_results(majority_x_correct_prediction_tables[class_id])
                results[f"metric_2.1_chi^2_p_value_{str(class_id).zfill(2)}"] = self.majority_x_class_prediction_chi2_metrics[class_id].mean()
                results[f"metric_2.1_spd_{str(class_id).zfill(2)}"] = self.majority_x_class_prediction_spd_metrics[class_id].mean()
                logger.info(f"Metric 2.1 Table for Class {str(class_id).zfill(2)}: chi^2 p-value = {self.majority_x_class_prediction_chi2_metrics[class_id].mean():.4f}")
                logger.info(f"Metric 2.1 Table for Class {str(class_id).zfill(2)}: SPD = {self.majority_x_class_prediction_spd_metrics[class_id].mean():.4f}")


            # filter predictions metric
            # Uses filter output and majority labels from TRAIN set
            if self.config["adhoc"].get("use_poison_filtering_defense", False):
                kept_mask = np.zeros_like(self.y_poison, dtype=np.bool_)
                kept_mask[self.indices_to_keep] = True
                kept_mask_unpoisoned = kept_mask[~poisoned_mask]
                y_unpoisoned_pred = self.model.predict(x_unpoisoned, **self.predict_kwargs).argmax(1)
                correct_prediction_mask_unpoisoned = (y_unpoisoned == y_unpoisoned_pred)

                majority_x_passed_filter_tables = metrics.make_contingency_tables(
                    y_unpoisoned,
                    majority_mask_unpoisoned,
                    kept_mask_unpoisoned
                    )

                for class_id in np.unique(self.y_clean):
                    
                    self.majority_x_passed_filter_chi2_metrics[class_id].add_results(majority_x_passed_filter_tables[class_id])
                    self.majority_x_passed_filter_spd_metrics[class_id].add_results(majority_x_passed_filter_tables[class_id])
                    results[f"metric_2.2_chi^2_p_value_{str(class_id).zfill(2)}"] = self.majority_x_passed_filter_chi2_metrics[class_id].mean()
                    results[f"metric_2.2_spd_{str(class_id).zfill(2)}"] = self.majority_x_passed_filter_spd_metrics[class_id].mean()
                    logger.info(f"Metric 2.2 Table for Class {str(class_id).zfill(2)}: chi^2 p-value = {self.majority_x_passed_filter_chi2_metrics[class_id].mean():.4f}")
                    logger.info(f"Metric 2.2 Table for Class {str(class_id).zfill(2)}: SPD = {self.majority_x_passed_filter_spd_metrics[class_id].mean():.4f}")
            

            if self.sample_exporter is not None:
                self.sample_exporter.export_data_per_example(majority_mask_test_set, "majority_flags")
        self.results = results
