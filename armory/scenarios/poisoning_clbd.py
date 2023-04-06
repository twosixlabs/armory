"""
Clean label backdoor poisoning scenario
"""

import copy

import numpy as np

from armory.logs import log
from armory.scenarios.poison import DatasetPoisoner, Poison
from armory.scenarios.utils import from_categorical, to_categorical
from armory.utils import config_loading


class CleanDatasetPoisoner:
    def __init__(self, attack, categorical=True):
        self.attack = attack
        self.categorical = bool(categorical)

    def poison_dataset(self, x, y, return_index=False):
        if self.categorical:
            y = to_categorical(y)

        x_poison, y_poison = self.attack.poison(x, y)

        if self.categorical:
            y_poison = from_categorical(y_poison)

        if return_index:
            poison_index = []
            if len(x) == len(x_poison):
                for i, (x_i, x_poison_i) in enumerate(zip(x, x_poison)):
                    if (x_i != x_poison_i).any():
                        poison_index.append(i)
            else:
                log.warning(
                    f"len(x_poison) {len(x_poison)} != len(x) {len(x)}. Returning []"
                )
            poison_index = np.array(poison_index)
            return x_poison, y_poison, poison_index
        return x_poison, y_poison


class Poison_CLBD(Poison):
    def load_poisoner(self):
        adhoc_config = self.config.get("adhoc") or {}
        attack_config = copy.deepcopy(self.config["attack"])
        if attack_config.get("type") == "preloaded":
            raise ValueError("preloaded attacks not currently supported for poisoning")

        self.use_poison = bool(adhoc_config["poison_dataset"])
        self.source_class = adhoc_config["source_class"]
        self.target_class = adhoc_config["target_class"]
        if self.use_poison:
            proxy, _ = config_loading.load_model(self.config["model"])

            fit_kwargs = {
                "batch_size": self.fit_batch_size,
                "nb_epochs": self.train_epochs,
            }
            log.info("Fitting proxy classifier...")
            if attack_config.get("use_adversarial_trainer"):
                from art.defences.trainer import AdversarialTrainerMadryPGD

                log.info("Using adversarial trainer...")
                trainer_kwargs = attack_config.pop("adversarial_trainer_kwargs", {})
                trainer_kwargs.update(fit_kwargs)
                trainer = AdversarialTrainerMadryPGD(proxy, **trainer_kwargs)
                trainer.fit(self.x_clean, self.y_clean)
                proxy = trainer.get_classifier()
            else:
                fit_kwargs.update(
                    dict(
                        shuffle=True,
                        verbose=False,
                    )
                )
                proxy.fit(self.x_clean, self.y_clean, **fit_kwargs)

            attack_config["kwargs"]["proxy_classifier"] = proxy

            attack, backdoor = config_loading.load(attack_config)
            self.poisoner = CleanDatasetPoisoner(attack)
            self.test_poisoner = DatasetPoisoner(
                backdoor,
                self.source_class,
                self.target_class,
            )
