import os
import copy

import numpy as np

from armory.scenarios.poison import Poison
from armory.utils.poisoning import FairnessMetrics
from armory.logs import log
from armory.utils import config_loading, metrics
from armory import paths


class DatasetPoisonerWitchesBrew:
    def __init__(
        self,
        attack,
        x_test,
        y_test,
        source_class,
        target_class,
        trigger_index,
        data_filepath,
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

        y_trigger = self.target_class

        poison_x, poison_y, poison_index = self.attack.poison(
            self.data_filepath,
            x_trigger,
            y_trigger,
            x_train,
            y_train,
            self.trigger_index,
        )

        if return_index:
            return poison_x, poison_y, poison_index
        return poison_x, poison_y


class WitchesBrewScenario(Poison):
    def load_poisoner(self):
        adhoc_config = self.config.get("adhoc") or {}
        attack_config = self.config["attack"]
        if attack_config.get("type") == "preloaded":
            raise ValueError("preloaded attacks not currently supported for poisoning")

        self.use_poison = bool(adhoc_config["poison_dataset"])

        dataset_config = self.config["dataset"]
        test_dataset = config_loading.load_dataset(
            dataset_config, split="test", num_batches=None, **self.dataset_kwargs
        )
        x_test, y_test = (np.concatenate(z, axis=0) for z in zip(*list(test_dataset)))

        # TODO Needs discussion -- how are we actually going to pick trigger images.
        trigger_index = adhoc_config["trigger_index"]
        if isinstance(trigger_index, int):
            trigger_index = [trigger_index]
        self.trigger_index = trigger_index

        target_class = adhoc_config["target_class"]
        if isinstance(target_class, int):
            target_class = [target_class] * len(self.trigger_index)
        if len(target_class) == 1:
            target_class = target_class * len(self.trigger_index)
        self.target_class = target_class

        source_class = adhoc_config["source_class"]
        if isinstance(source_class, int):
            source_class = [source_class] * len(self.trigger_index)
        if len(source_class) == 1:
            source_class = source_class * len(self.trigger_index)
        self.source_class = source_class

        if len(self.target_class) != len(self.trigger_index):
            raise ValueError(
                "target_class should have one element or be the same length as trigger_index"
            )

        if len(self.source_class) != len(self.trigger_index):
            raise ValueError(
                "source_class should have one element or be the same length as trigger_index"
            )

        for i, trigger_ind in enumerate(self.trigger_index):
            if y_test[trigger_ind] != self.source_class[i]:
                raise ValueError(
                    f"Trigger image {i} does not belong to source class (class {y_test[trigger_ind]} != class {self.source_class[i]})"
                )

        if sum([t==s for t, s in zip(self.target_class, self.source_class)]) > 0:
            raise ValueError(f" No target class may equal source class; got target = {self.target_class} and source = {self.source_class}")

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

            attack_dir = os.path.join(paths.runtime_paths().saved_model_dir, "attacks")
            os.makedirs(attack_dir, exist_ok=True)

            attack = config_loading.load_attack(attack_config, self.model)

            if data_filepath is not None:
                data_filepath = os.path.join(attack_dir, data_filepath)

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

        self.y_pred = y_pred  # for exporting when function returns

    def run_attack(self):
        # Only called for the trigger images

        x, y = self.x, self.y

        x.flags.writeable = False
        y_pred_adv = self.model.predict(x, **self.predict_kwargs)

        self.trigger_accuracy_metric.add_results(y, y_pred_adv)

        self.y_pred_adv = y_pred_adv  # for exporting when function returns

    def evaluate_current(self):

        if self.i in self.trigger_index:
            self.run_attack()
        else:
            self.run_benign()

        # TODO Needs discussion--what is useful to export.
        # This just exports clean test samples up to num_eval_batches, and all the triggers.
        # The use of sample_exporter is nonconventional since in this attack,
        # we don't have benign and adversarial versions of the same test image.
        if (
            self.num_export_batches > self.sample_exporter.saved_batches
        ) or self.i in self.trigger_index:
            if self.sample_exporter.saved_samples == 0:
                self.sample_exporter._make_output_dir()
            name = "trigger" if self.i in self.trigger_index else "non-trigger"
            self.sample_exporter._export_image(self.x[0], name=name)
            self.sample_exporter.saved_samples += 1
            self.sample_exporter.saved_batches = (
                self.sample_exporter.saved_samples
                // self.config["dataset"]["batch_size"]
            )

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

