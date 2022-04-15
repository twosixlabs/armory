import os
import copy

import numpy as np
from art.utils import to_categorical

from armory.scenarios.poison import Poison
from armory.utils.poisoning import FairnessMetrics
from armory.logs import log
from armory.utils import config_loading, metrics
from armory import paths



class DatasetPoisonerWitchesBrew:
    def __init__(
        self, attack, x_test, y_test, source_class, target_class, trigger_index, data_filepath
    ):
        """
        Individual source-class triggers are chosen from x_test.  At poison time, the
        train set is modified to induce misclassification of the triggers as target_class.

        """
        self.attack = attack
        self.x_test = x_test
        self.y_test = y_test
        self.source_class = source_class
        self.target_class = target_class
        self.trigger_index = trigger_index
        self.data_filepath = data_filepath

    def poison_dataset(self, x_train, y_train, return_index=True):
        """
        Return a poisoned version of dataset x, y
            if return_index, return x, y, index
        """
        if len(x_train) != len(y_train):
            raise ValueError("Sizes of x and y do not match")

        x_trigger = self.x_test[self.trigger_index]
        if len(x_trigger.shape) == 3:
            x_trigger = np.expand_dims(x_trigger, axis=0)

        y_trigger = [self.target_class] * len(self.trigger_index) # TODO assuming all images have same target.

        # TODO is armory data oriented and scaled right for art_experimental
        poison_x, poison_y, poison_index = self.attack.poison(
            self.data_filepath, x_trigger, y_trigger, x_train, y_train, self.trigger_index
        )

        if return_index:
            return poison_x, poison_y, poison_index
        return poison_x, poison_y


class CifarWitchesBrew(Poison):

    def load_poisoner(self):
        adhoc_config = self.config.get("adhoc") or {}
        attack_config = self.config["attack"]
        if attack_config.get("type") == "preloaded":
            raise ValueError("preloaded attacks not currently supported for poisoning")

        self.use_poison = bool(adhoc_config["poison_dataset"])
        self.source_class = adhoc_config["source_class"]
        self.target_class = adhoc_config["target_class"]

        dataset_config = self.config["dataset"]
        test_dataset = config_loading.load_dataset(
            dataset_config, split="test", num_batches=None, **self.dataset_kwargs
        )
        x_test, y_test = (np.concatenate(z, axis=0) for z in zip(*list(test_dataset)))

        # TODO how to pick or set trigger images
        trigger_index = adhoc_config["trigger_index"]
        if isinstance(trigger_index, int):
            trigger_index = [trigger_index]
        self.trigger_index = trigger_index

        print (np.where(y_test == self.source_class)[0][:15])

        for i in self.trigger_index:
            if y_test[i] != self.source_class:
                raise ValueError(
                    f"Trigger image {i} does not belong to source class (class {y_test[i]} != class {self.source_class})"
                )

        if self.use_poison:

            attack_config["kwargs"]["percent_poison"] = adhoc_config[
                "fraction_poisoned"
            ]
            attack_config["kwargs"]["source_class"] = self.source_class
            attack_config["kwargs"]["target_class"] = self.target_class

            data_filepath = (
                attack_config["kwargs"].pop("data_filepath")
                if "data_filepath" in attack_config["kwargs"].keys()
                else None
            )

            attack = config_loading.load_attack(attack_config, self.model)
            if data_filepath is not None:
                data_filepath = os.path.join(
                    paths.runtime_paths().dataset_dir, data_filepath
                )
            self.poisoner = DatasetPoisonerWitchesBrew(
                attack,
                x_test,
                y_test,
                self.source_class,
                self.target_class,
                self.trigger_index,
                data_filepath,
            )
            self.test_poisoner = self.poisoner


    def load_metrics(self):
        self.non_trigger_accuracy_metric = metrics.MetricList("categorical_accuracy")
        self.trigger_accuracy_metric = metrics.MetricList("categorical_accuracy")

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

    # TODO think about sample exporting?


    def load_dataset(self, eval_split_default="test"):
        # Over-ridden because we need batch_size = 1 for the test set for this attack.

        dataset_config = self.config["dataset"]
        dataset_config = copy.deepcopy(dataset_config)
        dataset_config["batch_size"] = 1
        eval_split = dataset_config.get("eval_split", eval_split_default)
        log.info(f"Loading test dataset {dataset_config['name']}...")
        self.test_dataset = config_loading.load_dataset(
            dataset_config,
            split=eval_split,
            num_batches=self.num_eval_batches,
            **self.dataset_kwargs,
        )
        self.i = -1

    def run_benign(self):
        # Called for all non-triggers

        x, y = self.x, self.y

        x.flags.writeable = False
        y_pred = self.model.predict(x, **self.predict_kwargs)

        self.non_trigger_accuracy_metric.add_results(y, y_pred)

        for y_, y_pred_ in zip(y, y_pred):
            if y_ not in self.benign_test_accuracy_per_class.keys():
                self.benign_test_accuracy_per_class[y_] = []

            self.benign_test_accuracy_per_class[y_].append(
                y_ == np.argmax(y_pred_, axis=-1)
            )

    def run_attack(self):
        # Only called for the trigger images

        x, y = self.x, self.y

        x.flags.writeable = False
        y_pred_adv = self.model.predict(x, **self.predict_kwargs)

        self.trigger_accuracy_metric.add_results(y, y_pred_adv)

    def evaluate_current(self):

        if self.i in self.trigger_index:
            self.run_attack()
        else:
            self.run_benign()

    def _add_accuracy_metrics_results(self):
        """ Adds accuracy results for trigger and non-trigger images
        """
        self.results[
            "accuracy_non_trigger_images"
        ] = self.non_trigger_accuracy_metric.mean()
        log.info(
            f"Accuracy on non-trigger images: {self.non_trigger_accuracy_metric.mean():.2%}"
        )

        self.results["accuracy_trigger_images"] = self.trigger_accuracy_metric.mean()
        log.info(
            f"Accuracy on trigger images: {self.trigger_accuracy_metric.mean():.2%}"
        )

    def finalize_results(self):
        self.results = {}

        self._add_accuracy_metrics_results()

        self._add_supplementary_metrics_results()

        if self.use_filtering_defense:
            self._add_filter_metrics_results()

        if hasattr(self, "fairness_metrics") and not self.check_run:
            self._add_fairness_metrics_results()
