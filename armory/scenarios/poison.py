"""
Extended scenario for poisoning
"""

import copy
from typing import Optional
import os
import random

import numpy as np

from armory.utils.poisoning import FairnessMetrics
from armory.utils.export import ImageClassificationExporter
from armory.scenarios.scenario import Scenario
from armory.scenarios.utils import to_categorical
from armory.utils import config_loading, metrics
from armory.logs import log


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
            log.warning(f"0 of {total} poisoned for class {self.source_class}.")
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
                log.info(
                    f"model.preprocessing_defences: {model.preprocessing_defences}"
                )
        self.model = model
        self.predict_kwargs = model_config.get("predict_kwargs", {})
        self.use_filtering_defense = self.config["adhoc"].get(
            "use_poison_filtering_defense", False
        )

    def set_dataset_kwargs(self):
        self.dataset_kwargs = dict(epochs=1, shuffle_files=False)

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
        log.info(f"Loading dataset {dataset_config['name']}...")
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
            adhoc_config.get("use_poison_filtering_defense", True)
            and not self.check_run
        ):
            defense_config = copy.deepcopy(self.config["defense"] or {})
            if "data_augmentation" in defense_config:
                defense_config.pop("data_augmentation")  # NOTE: RESISC10 ONLY

            # Assumes classifier_for_defense and classifier use same preprocessing function
            defense_model_config = adhoc_config.get(
                "defense_model", self.config["model"]
            )
            classifier_for_defense, _ = config_loading.load_model(defense_model_config)
            log.info(
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

            detection_kwargs = defense_config.get("kwargs", {})
            _, is_clean = defense.detect_poison(**detection_kwargs)
            is_clean = np.array(is_clean)
            log.info(f"Total clean data points: {np.sum(is_clean)}")
            is_dirty = is_clean.astype(np.int64) == 0
            log.info(f"Total dirty data points: {np.sum(is_dirty)}")

            log.info("Filtering out detected poisoned samples")
            indices_to_keep = is_clean == 1

        else:
            log.info(
                "Defense does not require filtering. Model fitting will use all data."
            )
            indices_to_keep = np.ones_like(self.y_poison, dtype=np.bool_)

        self.x_train = self.x_poison[indices_to_keep]
        self.y_train = self.y_poison[indices_to_keep]
        self.indices_to_keep = indices_to_keep

    def fit(self):
        if len(self.x_train):
            log.info("Fitting model")
            self.model.fit(
                self.x_train,
                self.label_function(self.y_train),
                batch_size=self.fit_batch_size,
                nb_epochs=self.train_epochs,
                verbose=False,
                shuffle=True,
            )
        else:
            log.warning("All data points filtered by defense. Skipping training")

    def load_attack(self):
        raise NotImplementedError(
            "Not implemented for poisoning scenario. Use load_poisoner"
        )

    def load_dataset(self, eval_split_default="test"):
        dataset_config = self.config["dataset"]
        eval_split = dataset_config.get("eval_split", eval_split_default)
        # Evaluate the ART model on benign test examples
        log.info(f"Loading test dataset {dataset_config['name']}...")
        self.test_dataset = config_loading.load_dataset(
            dataset_config,
            split=eval_split,
            num_batches=self.num_eval_batches,
            **self.dataset_kwargs,
        )
        self.i = -1

    def load_metrics(self):
        self.accuracy_on_benign_data_all_classes = metrics.MetricList(
            "categorical_accuracy"
        )
        self.accuracy_on_benign_data_source_class = metrics.MetricList(
            "categorical_accuracy"
        )
        if self.use_poison:
            self.accuracy_on_poisoned_data_all_classes = metrics.MetricList(
                "categorical_accuracy"
            )
            self.attack_success_rate = metrics.MetricList("categorical_accuracy")
            # attack_success_rate is not just 1 - (accuracy on poisoned source class)
            # because it only counts examples misclassified as target, and no others.

        self.benign_test_accuracy_per_class = (
            {}
        )  # store accuracy results for each class
        if self.config["adhoc"].get("compute_fairness_metrics", False):
            self.fairness_metrics = FairnessMetrics(
                self.config["adhoc"], self.use_filtering_defense, self
            )
        else:
            log.warning(
                "Not computing fairness metrics.  If these are desired, set 'compute_fairness_metrics':true under the 'adhoc' section of the config"
            )

    def _load_sample_exporter(self):
        return ImageClassificationExporter(self.scenario_output_dir)

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
        self.load_sample_exporter()

    def run_benign(self):
        x, y = self.x, self.y

        x.flags.writeable = False
        y_pred = self.model.predict(x, **self.predict_kwargs)

        self.accuracy_on_benign_data_all_classes.add_results(y, y_pred)
        source = y == self.source_class
        # NOTE: uses source->target trigger
        if source.any():
            self.accuracy_on_benign_data_source_class.add_results(
                y[source], y_pred[source]
            )

        self.y_pred = y_pred
        self.source = source

        for y_, y_pred_ in zip(y, y_pred):
            if y_ not in self.benign_test_accuracy_per_class.keys():
                self.benign_test_accuracy_per_class[y_] = []

            self.benign_test_accuracy_per_class[y_].append(
                y_ == np.argmax(y_pred_, axis=-1)
            )

    def run_attack(self):
        x, y = self.x, self.y
        source = self.source

        x_adv, _ = self.test_poisoner.poison_dataset(x, y, fraction=1.0)
        x_adv.flags.writeable = False
        y_pred_adv = self.model.predict(x_adv, **self.predict_kwargs)

        self.accuracy_on_poisoned_data_all_classes.add_results(y, y_pred_adv)
        # NOTE: uses source->target trigger
        if source.any():
            self.attack_success_rate.add_results(
                [self.target_class] * source.sum(), y_pred_adv[source]
            )  # counts number of source images classified as target

        self.x_adv = x_adv
        self.y_pred_adv = y_pred_adv

    def evaluate_current(self):
        self.run_benign()
        if self.use_poison:
            self.run_attack()

        if self.num_export_batches > self.sample_exporter.saved_batches:
            # Note: Will still output x_adv even if it is the same as x, i.e. not poisoned
            self.sample_exporter.export(
                x=self.x,
                x_adv=self.x_adv,
                y=self.y,
                y_pred_clean=self.y_pred,
                y_pred_adv=self.y_pred_adv,
            )

    def _add_filter_metrics_results(self):
        """ Adds filter-specific metrics to self.results:
        Number of samples removed total and per class, true and false positives, F1 score
        """

        removed = (1 - self.indices_to_keep).astype(np.bool)
        poisoned = np.zeros_like(self.y_clean).astype(np.bool)
        poisoned[self.poison_index.astype(np.int64)] = True

        false_negatives = int(np.sum(~removed & poisoned))
        true_positives = int(np.sum(removed & poisoned))
        true_negatives = int(np.sum(~removed & ~poisoned))
        false_positives = int(np.sum(removed & ~poisoned))

        false_negative_rate = (
            0 if self.n_poisoned == 0 else false_negatives / self.n_poisoned
        )
        true_positive_rate = (
            0 if self.n_poisoned == 0 else true_positives / self.n_poisoned
        )
        true_negative_rate = true_negatives / self.n_clean
        false_positive_rate = false_positives / self.n_clean

        f1_score = true_positives / (
            true_positives + 0.5 * (false_positives + false_negatives)
        )

        self.results["filter_true_positives"] = true_positives
        self.results["filter_false_positives"] = false_positives
        self.results["filter_true_negatives"] = true_negatives
        self.results["filter_false_negatives"] = false_negatives
        self.results["filter_true_positive_rate"] = true_positive_rate
        self.results["filter_false_positive_rate"] = false_positive_rate
        self.results["filter_true_negative_rate"] = true_negative_rate
        self.results["filter_false_negative_rate"] = false_negative_rate
        self.results["filter_f1_score"] = f1_score
        self.results["filter_fraction_data_removed"] = removed.mean()
        self.results["filter_N_samples_removed"] = int(removed.sum())

        for y in self.train_set_class_labels:
            self.results[f"class_{y}_N_train_samples_removed"] = int(
                np.sum(self.y_clean[removed] == y)
            )

    def _add_fairness_metrics_results(self):
        """ Adds fairness metrics to self.results:
            model bias and filter bias on class subpopulations
        """

        # Get unpoisoned test set
        dataset_config = self.config["dataset"]
        test_dataset = config_loading.load_dataset(
            dataset_config, split="test", num_batches=None, **self.dataset_kwargs
        )

        # The following functions will add data to self.results
        log_lines = self.fairness_metrics.add_cluster_metrics(
            self.x_poison,
            self.y_poison,
            self.poison_index,
            self.indices_to_keep,
            test_dataset,
            self.train_set_class_labels,
            self.test_set_class_labels,
        )
        for line in log_lines:
            log.info(line)
        if self.use_filtering_defense:
            log_lines = self.fairness_metrics.add_filter_perplexity(
                self.y_clean, self.poison_index, self.indices_to_keep
            )
            for line in log_lines:
                log.info(line)

    def _add_accuracy_metrics_results(self):
        """ Adds the main accuracy results to self.results:
            poisoned and benign performance on whole test set and on source class
        """
        self.results[
            "accuracy_on_benign_data_all_classes"
        ] = self.accuracy_on_benign_data_all_classes.mean()
        self.results[
            "accuracy_on_benign_data_source_class"
        ] = self.accuracy_on_benign_data_source_class.mean()
        log.info(
            f"Accuracy on benign data--all classes: {self.accuracy_on_benign_data_all_classes.mean():.2%}"
        )
        log.info(
            f"Accuracy on benign data--source class: {self.accuracy_on_benign_data_source_class.mean():.2%}"
        )

        if self.use_poison:
            self.results[
                "accuracy_on_poisoned_data_all_classes"
            ] = self.accuracy_on_poisoned_data_all_classes.mean()
            self.results["attack_success_rate"] = self.attack_success_rate.mean()
            log.info(
                f"Accuracy on poisoned data--all classes: {self.accuracy_on_poisoned_data_all_classes.mean():.2%}"
            )
            log.info(
                f"Attack success rate: {self.attack_success_rate.mean():.2%}"
            )  # percent of poisoned source examples that get classified as target

    def _add_supplementary_metrics_results(self):
        """ Adds additional metrics  to self.results:
        N poisoned, N clean, N samples per class, benign accuracy per class
        """
        self.n_poisoned = int(len(self.poison_index))
        self.n_clean = (
            len(self.y_clean) - self.n_poisoned
        )  # self.y_clean is the whole pre-poison train set
        self.results["N_poisoned_train_samples"] = self.n_poisoned
        self.results["N_clean_train_samples"] = self.n_clean

        self.train_set_class_labels = sorted(list(np.unique(self.y_clean)))
        self.test_set_class_labels = sorted(
            list(self.benign_test_accuracy_per_class.keys())
        )
        if self.test_set_class_labels != self.train_set_class_labels:
            log.warning(
                "Test set contains a strict subset of train set classes.  Some metrics for missing classes may not be computed."
            )
        for y in self.train_set_class_labels:
            self.results[f"class_{y}_N_train_samples"] = int(np.sum(self.y_clean == y))
        for y in self.test_set_class_labels:
            self.results[f"class_{y}_unpoisoned_test_accuracy"] = np.mean(
                self.benign_test_accuracy_per_class[y]
            )

    def finalize_results(self):
        self.results = {}

        self._add_accuracy_metrics_results()

        self._add_supplementary_metrics_results()

        if self.use_filtering_defense:
            self._add_filter_metrics_results()

        if hasattr(self, "fairness_metrics") and not self.check_run:
            self._add_fairness_metrics_results()
