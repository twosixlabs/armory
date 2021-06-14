"""
Extended scenario for poisoning
"""

import copy
import logging
from typing import Optional
import os
import random

import numpy as np

from tqdm import tqdm

from armory.scenarios.scenario import Scenario
from armory.scenarios.utils import to_categorical
from armory.utils import config_loading, metrics

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
        poison_x, poison_y = np.array(poison_x), np.array(poison_y)

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

    def set_random_seed(self):
        # Set random seed due to large variance in attack and defense success
        self.seed = self.config["adhoc"]["split_id"]
        np.random.seed(self.seed)
        random.seed(self.seed)

    def load_model(self):
        if self.config["sysconfig"].get("use_gpu"):
            os.environ["TF_CUDNN_DETERMINISM"] = "1"

        config_adhoc = self.config.get("adhoc") or {}
        self.source_class = config_adhoc["source_class"]
        self.target_class = config_adhoc["target_class"]
        self.train_epochs = config_adhoc["train_epochs"]
        self.fit_batch_size = config_adhoc.get(
            "fit_batch_size", self.config["dataset"]["batch_size"]
        )

        # Scenario assumes canonical preprocessing_fn is used makes images all same size
        self.estimator, _ = config_loading.load_model(self.config["model"])

        # Flag to determine whether training uses categorical or sparse labels
        self.categorical_labels = config_adhoc.get("defense_categorical_labels", True)
        if self.categorical_labels:
            self.label_function = to_categorical
        else:
            self.label_function = lambda y: y

    def load_defense(self):
        pass

    def set_dataset_kwargs(self):
        dataset_config = self.config["dataset"]
        self.dataset_kwargs = dict(epochs=1, shuffle_files=False,)
        self.train_split = dataset_config.get("train_split", "train")
        self.validation_split = dataset_config.get("eval_split", "test")
        self.test_split = dataset_config.get("eval_split", "test")

    def load_train_dataset(self):
        """
        Load and create in memory dataset
            detect_poison does not currently support data generators
        """
        dataset_config = self.config["dataset"]
        self.set_dataset_kwargs()
        logger.info(f"Loading dataset {dataset_config['name']}...")
        ds = config_loading.load_dataset(
            dataset_config, split=self.train_split, **self.dataset_kwargs
        )
        self.x_clean, self.y_clean = (np.concatenate(z, axis=0) for z in zip(*list(ds)))

    def load_attack(self):
        adhoc_config = self.config["adhoc"]
        attack_config = self.config["attack"]
        if attack_config.get("type") == "preloaded":
            raise ValueError("preloaded attacks not currently supported for poisoning")

        self.use_poison = bool(adhoc_config["poison_dataset"])
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
            ) = self.poisoner.poison_datset(
                self.x_clean, self.y_clean, return_index=True
            )
        else:
            self.x_poison, self.y_poison, self.poison_index = (
                self.x_clean,
                self.y_clean,
                np.array([]),
            )

    def filter_dataset(self):
        config_adhoc = self.config["adhoc"]
        # filtering defense requires more than a single batch to run properly
        if (
            config_adhoc.get("use_poison_filtering_defense", True)
            and not self.check_run
        ):
            defense_config = copy.deepcopy(self.config["defense"] or {})
            if "data_augmentation" in defense_config:
                defense_config.pop("data_augmentation")  # NOTE: RESISC10 ONLY

            # Assumes classifier_for_defense and classifier use same preprocessing function
            defense_model_config = config_adhoc.get(
                "defense_model", self.config["model"]
            )
            classifier_for_defense, _ = config_loading.load_model(defense_model_config)
            logger.info(
                f"Fitting model {defense_model_config['module']}.{defense_model_config['name']} "
                f"for defense {defense_config['name']}..."
            )
            # Flag to determine whether defense_classifier is trained directly
            #     (default API) or is trained as part of detect_poisons method
            if config_adhoc.get("fit_defense_classifier_outside_defense", True):
                classifier_for_defense.fit(
                    self.x_poison,
                    self.label_function(self.y_poison),
                    batch_size=self.fit_batch_size,
                    nb_epochs=config_adhoc.get(
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

            detection_kwargs = config_adhoc.get("detection_kwargs", {})
            _, is_clean = defense.detect_poison(**detection_kwargs)
            is_clean = np.array(is_clean)
            logger.info(f"Total clean data points: {np.sum(is_clean)}")

            logger.info("Filtering out detected poisoned samples")
            indices_to_keep = is_clean == 1  # TODO: redundant?

        else:
            logger.info(
                "Defense does not require filtering. Model fitting will use all data."
            )
            indices_to_keep = ...

        # TODO: measure TP and FP rates for filtering
        self.x_train = self.x_poison[indices_to_keep]
        self.y_train = self.y_poison[indices_to_keep]
        self.indices_to_keep = indices_to_keep

    def fit_model(self):
        if len(self.x_train):
            logger.info("Fitting model")
            self.estimator.fit(
                self.x_train,
                self.label_function(self.y_train),
                batch_size=self.fit_batch_size,
                nb_epochs=self.train_epochs,
                verbose=False,
                shuffle=True,
            )
        else:
            logger.warning("All data points filtered by defense. Skipping training")

    def load_metrics(self):
        self.benign_validation_metric = metrics.MetricList("categorical_accuracy")
        self.target_class_benign_metric = metrics.MetricList("categorical_accuracy")
        if self.use_poison:
            self.poisoned_test_metric = metrics.MetricList("categorical_accuracy")
            self.poisoned_targeted_test_metric = metrics.MetricList(
                "categorical_accuracy"
            )

    def validate(self):
        logger.info("Validating on clean validation data")
        val_data = config_loading.load_dataset(
            self.config["dataset"], split=self.validation_split, **self.dataset_kwargs
        )
        for x, y in tqdm(val_data, desc="Validation"):
            # Ensure that input sample isn't overwritten by classifier
            x.flags.writeable = False
            y_pred = self.estimator.predict(x)
            self.benign_validation_metric.add_results(y, y_pred)
            y_pred_target_class = y_pred[y == self.source_class]
            if len(y_pred_target_class):
                self.target_class_benign_metric.add_results(
                    [self.source_class] * len(y_pred_target_class), y_pred_target_class
                )

    def test(self):
        if self.use_poison:
            logger.info("Testing on poisoned test data")
            test_data = config_loading.load_dataset(
                self.config["dataset"], split=self.test_split, **self.dataset_kwargs
            )
            for x, y in tqdm(test_data, desc="Testing"):
                x, _ = self.test_poisoner.poison_dataset(x, y, fraction=1.0)
                # Ensure that input sample isn't overwritten by classifier
                x.flags.writeable = False
                y_pred = self.estimator.predict(x)
                self.poisoned_test_metric.add_results(y, y_pred)

                y_pred_targeted = y_pred[y == self.source_class]
                if len(y_pred_targeted):
                    self.poisoned_targeted_test_metric.add_results(
                        [self.target_class] * len(y_pred_targeted), y_pred_targeted
                    )

    def finalize(self):
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
        self.results = results

    def _evaluate(self) -> dict:
        """
        Evaluate a config file for classification robustness against attack.
        """
        self.set_random_seed()
        self.load_model()
        self.load_defense()
        self.load_attack()
        self.load_train_dataset()
        self.poison_dataset()
        self.filter_dataset()
        self.load_metrics()
        self.fit_model()
        self.validate()
        self.test()
        self.finalize()
        return self.results

    def load(self):
        pass

    # TODO: perhaps it would be better to create a simpler common base class?

    def _load_estimator(self):
        raise NotImplementedError("Not implemented for poisoning scenario")

    def _load_defense(self, estimator, train_split_default="train"):
        raise NotImplementedError("Not implemented for poisoning scenario")

    def load_dataset(self):
        raise NotImplementedError("Not implemented for poisoning scenario")

    def evaluate_all(self):
        raise NotImplementedError("Not implemented for poisoning scenario")

    def next(self):
        raise NotImplementedError("Not implemented for poisoning scenario")

    def benign(self):
        raise NotImplementedError("Not implemented for poisoning scenario")

    def adversary(self):
        raise NotImplementedError("Not implemented for poisoning scenario")

    def evaluate_current(self):
        raise NotImplementedError("Not implemented for poisoning scenario")
