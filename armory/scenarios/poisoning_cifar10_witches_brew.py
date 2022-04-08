from armory.scenarios.poison import Poison
from armory.logs import log
from armory.utils import config_loading
import numpy as np


class DatasetPoisonerWitchesBrew():
    def __init__(self, attack, source_class, target_class, x_test, y_test):
        """
        Individual source-class triggers are chosen from x_test.  At poison time, the
        train set is modified to induce misclassification of the triggers as target_class.

        """
        self.attack = attack
        self.source_class = source_class
        self.target_class = target_class
        self.x_test = x_test
        self.y_test = y_test


    def poison_dataset(self, x_train, y_train, return_index=True):
        """
        Return a poisoned version of dataset x, y
            if return_index, return x, y, index
        If fraction is not None, use it to override default
        """
        if len(x_train) != len(y_train):
            raise ValueError("Sizes of x and y do not match")

        from art.utils import to_categorical
        print(self.y_test[:10])
        print(type(self.y_test))
        index_target = np.where(self.y_test==self.source_class)[0][5] # TODO index 0 because where returns 1-length tuple.  5 to just pick one.

        # Trigger sample
        x_trigger = self.x_test[index_target:index_target+1]
        y_trigger  = to_categorical([self.target_class], nb_classes=10)
        print ("HELLO FROM OVER HERE")
        print(index_target)
        

        filepath = "stuff.npz" # TODO where to get this from

        poison_x, poison_y, poison_index = self.attack.poison(filepath, x_trigger, y_trigger, x_train, y_train) # TODO is armory data oriented and scaled right for art

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
            dataset_config, split="test", num_batches=None, **self.dataset_kwargs,
        )
        x_test, y_test = (np.concatenate(z, axis=0) for z in zip(*list(test_dataset)))
        if self.use_poison:
            attack_config["kwargs"]["estimator"] = self.model
            attack_config["kwargs"]["percent_poison"] = adhoc_config["fraction_poisoned"]
            attack = config_loading.load(attack_config)
            self.poisoner = DatasetPoisonerWitchesBrew(
                attack,
                self.source_class,
                self.target_class,
                x_test,
                y_test,
            )
            self.test_poisoner = self.poisoner


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


    def evaluate_current(self):

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


