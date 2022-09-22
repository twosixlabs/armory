import copy

import numpy as np
from PIL import Image

from armory.scenarios.poison import Poison
from armory.logs import log
from armory.utils import config_loading
from armory.scenarios.utils import from_categorical


class DatasetPoisonerSleeperAgent:
    def __init__(self, attack, x_trigger, y_trigger, x_test, y_test):
        """
        Test images are preselected randomly from source class (or use all of them).
        These are caled x_trigger, with labels y_trigger.

        The train set is poisoned through gradient matching, and
        a patch is applied to the test images in x_trigger.

        Original x_test and y_test are provided for optional model
        retraining during the attack.

        """

        self.attack = attack
        self.x_trigger = x_trigger
        self.y_trigger = y_trigger
        self.x_test = x_test
        self.y_test = y_test

    def poison_dataset(self, x, y, return_index=True, fraction=None):
        # Needs fraction kwarg to be callable by inherited run_attack function in poison.py.
        # Here we just use it to signal we are poisoning the test set not the train set.

        x = copy.deepcopy(x)
        y = copy.deepcopy(y)  # attack modifies in place.  don't overwrite clean data

        if fraction == 1:
            # During the run_attack/run_benign cycle, poison test samples by adding patch
            for i in range(len(y)):
                if y[i] == self.attack.class_source:
                    x[i] = self.attack._apply_trigger_patch(np.expand_dims(x[i], 0))[0]
            return x, y

        # otherwise, poison the whole trian set

        x_poison, y_poison = self.attack.poison(
            self.x_trigger, self.y_trigger, x, y, self.x_test, self.y_test
        )
        # I think x_trigger is test images to poison in-place.  x_test and y_test are for model retraining.

        poison_index = self.attack.get_poison_indices()

        return x_poison, from_categorical(y_poison), poison_index


class SleeperAgentScenario(Poison):
    def load_poisoner(self):
        adhoc_config = self.config.get("adhoc") or {}
        attack_config = self.config["attack"]
        if attack_config.get("type") == "preloaded":
            raise ValueError("preloaded attacks not currently supported for poisoning")

        self.use_poison = bool(adhoc_config["poison_dataset"])
        self.source_class = adhoc_config["source_class"]
        self.target_class = adhoc_config["target_class"]

        if self.use_poison:

            #  Create and train proxy model for gradient matching attack.
            #  TODO Sleeper Agent will replace this with its own model.  Waiting on a fix from Shriti
            proxy_config = copy.deepcopy(self.config["model"])
            proxy_model, _ = config_loading.load_model(proxy_config)
            log.info("Fitting proxy model for attack . . .")
            proxy_model.fit(
                self.x_clean,
                self.label_function(self.y_clean),
                batch_size=self.fit_batch_size,
                nb_epochs=self.train_epochs,
                verbose=False,
                shuffle=True,
            )

            # load copy of test dataset for attack
            dataset_config = self.config["dataset"]
            test_dataset = config_loading.load_dataset(
                dataset_config, split="test", num_batches=None, **self.dataset_kwargs
            )
            x_test, y_test = (
                np.concatenate(z, axis=0) for z in zip(*list(test_dataset))
            )
            K = sum(
                y_test == self.source_class
            )  # number of source-class test images to use in x_trigger
            # K = 100  Not sure how many x_triggers we need to initialize attack, or why the attack needs them now;
            # but Shriti confirmed I don't need to save them and can re-add the patch
            # to any number of test images at eval time

            # Set additional attack config kwargs
            attack_config["kwargs"]["indices_target"] = np.asarray(
                self.y_clean == self.target_class
            ).nonzero()[
                0
            ]  # which train images belong to target class
            attack_config["kwargs"]["percent_poison"] = adhoc_config[
                "fraction_poisoned"
            ]
            attack_config["kwargs"]["class_source"] = self.source_class
            attack_config["kwargs"]["class_target"] = self.target_class
            attack_config["kwargs"]["learning_rate_schedule"] = tuple(
                attack_config["kwargs"]["learning_rate_schedule"]
            )  # convert to tuple as required by ART attack
            patch_size = attack_config["kwargs"].pop("patch_size")
            patch = Image.open(
                "armory/utils/triggers/" + attack_config["kwargs"]["patch"]
            )
            patch = np.asarray(patch.resize((patch_size, patch_size)))
            attack_config["kwargs"]["patch"] = patch

            # Create attack and pass it to DatasetPoisoner
            attack = config_loading.load_attack(attack_config, proxy_model)
            x_trigger = copy.deepcopy(x_test[y_test == self.source_class])[
                :K
            ]  # copy of test images to poison, because attack modifies these in place
            y_trigger = np.array(
                [self.target_class] * K
            )  # target labels to classify test images into
            N_class = len(np.unique(self.y_clean))

            self.poisoner = DatasetPoisonerSleeperAgent(
                attack,
                x_trigger,
                self.label_function(y_trigger, num_classes=N_class),
                x_test,  # test set passed for optional retraining of proxy model .
                self.label_function(y_test, num_classes=N_class),
            )
            self.test_poisoner = self.poisoner

            # Right now this is set up to discard the now-patched x_trigger (test images)
            # and patch them again during the run_benign/run_attack cycle.  This fits with existing
            # armory functions better and reduces the amount of functions I have to override

    def poison_dataset(self):
        self.hub.set_context(stage="poison")
        if self.use_poison:
            (
                self.x_poison,
                self.y_poison,
                self.poison_index,
            ) = self.poisoner.poison_dataset(
                self.x_clean,
                self.label_function(self.y_clean),
                return_index=True,
            )
        else:
            self.x_poison, self.y_poison, self.poison_index = (
                self.x_clean,
                self.y_clean,
                np.array([]),
            )

        self.record_poison_and_data_info()
