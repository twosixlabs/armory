import copy
import logging

from armory.utils import config_loading
from armory.scenarios.poison import Poison, DatasetPoisoner

# from armory.scenarios.poison import to_categorical

logger = logging.getLogger(__name__)


class CleanDatasetPoisoner:
    def __init__(self, attack):  # , categorical_labels=False):
        self.attack = attack
        # self.categorical_labels = bool(categorical_labels)

    def poison_dataset(self, x, y, return_index=False):
        if return_index:
            raise NotImplementedError("Not currently implemented")

        # if self.categorical_labels:  # TODO: Necessary?
        #     y = to_categorical(y)

        x_poison, y_poison = self.attack.poison(x, y)

        # if self.categorical_labels:  # TODO: Necessary?
        #     y_poison = np.argmax(y_poison, axis=1)

        return x_poison, y_poison


class Clean(Poison):
    def load_attack(self):
        adhoc_config = self.config["adhoc"]
        attack_config = copy.deepcopy(self.config["attack"])
        if attack_config.get("type") == "preloaded":
            raise ValueError("preloaded attacks not currently supported for poisoning")

        self.use_poison = bool(adhoc_config["poison_dataset"])
        if self.use_poison:
            proxy, _ = config_loading.load_model(self.config["model"])

            fit_kwargs = {
                "batch_size": self.fit_batch_size,
                "nb_epochs": self.train_epochs,
            }
            logger.info("Fitting proxy classifier...")
            if attack_config.get("use_adversarial_trainer"):
                from art.defences.trainer import AdversarialTrainerMadryPGD

                logger.info("Using adversarial trainer...")
                trainer_kwargs = attack_config.pop("adversarial_trainer_kwargs", {})
                trainer_kwargs.update(fit_kwargs)
                trainer = AdversarialTrainerMadryPGD(proxy, **trainer_kwargs)
                trainer.fit(self.x_clean, self.y_clean)
                proxy = trainer.get_classifier()
            else:
                fit_kwargs.update(dict(shuffle=True, verbose=False,))
                proxy.fit(self.x_clean, self.y_clean, **fit_kwargs)

            attack_config["kwargs"]["proxy_classifier"] = proxy

            attack, backdoor = config_loading.load(attack_config)
            self.poisoner = CleanDatasetPoisoner(attack)
            self.test_poisoner = DatasetPoisoner(
                backdoor, self.source_class, self.target_class,
            )
