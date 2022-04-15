import os

import numpy as np

from art.attacks.poisoning.gradient_matching_attack import GradientMatchingAttack
from armory.logs import log


class GradientMatchingWrapper(GradientMatchingAttack):
    def __init__(self, classifier, **kwargs):
        self.attack_kwargs = kwargs
        self.source_class = kwargs.pop("source_class")
        self.target_class = kwargs.pop("target_class")
        learning_rate_schedule = tuple(kwargs.pop("learning_rate_schedule"))

        super().__init__(
            classifier=classifier,
            # clip_values=(min_,max_),  get this somewhere
            learning_rate_schedule=learning_rate_schedule,
            **kwargs,
        )

    def poison(self, filepath, x_trigger, y_trigger, x_train, y_train, trigger_index):
        """
        Return a dataset poisoned to cause misclassification of x_trigger.

        filepath, string: path to load data from (if it exists) or save it to (if it doesn't)
        x_trigger, array: Images from the test set that we hope to misclassify
        y_trigger, array: Labels for x_trigger
        x_train, array: The training images to be poisoned
        y_train, array: Labels for x_train
        trigger_index, int or array: Index or indices of trigger images
        """

        # check if pre-poisoned dataset exists
        if filepath is not None and os.path.exists(filepath):
            log.info(f"Loading existing poisoned dataset at {filepath}")
            poison_npz = np.load(filepath)
            x_poison, y_poison, poison_index, load_trigger_index = (
                poison_npz["x_poison"],
                poison_npz["y_poison"],
                poison_npz["poison_index"],
                poison_npz["trigger_index"],
            )
            target_class, percent_poison, epsilon = (
                poison_npz["target_class"],
                poison_npz["percent_poison"],
                poison_npz["epsilon"],
            )

            if sorted(load_trigger_index) != sorted(trigger_index):
                raise ValueError(
                    f"Adversarial dataset at filepath {filepath} was trained for trigger image(s) {load_trigger_index}, not {trigger_index} as requested by the config."
                )
            # target class must match element-wise, can't sort
            if not (
                len(target_class) == len(self.target_class)
                and sum([t1 == t2 for t1, t2 in zip(target_class, self.target_class)])
                == len(target_class)
            ):
                raise ValueError(
                    f"Adversarial dataset at filepath {filepath} was trained for target class(es) {target_class}, not {self.target_class} as requested by the config."
                )
            if percent_poison != self.percent_poison:
                raise ValueError(
                    f"Adversarial dataset at filepath {filepath} has a poison frequency of {percent_poison}, not {self.percent_poison} as requested by the config."
                )
            if epsilon != self.epsilon:
                raise ValueError(
                    f"Adversarial dataset at filepath {filepath} has a L_inf perturbation bound of {epsilon}, not {self.epsilon} as requested by the config."
                )
            # No need to verify source class, since it will match if the trigger_index matched

        else:
            # Generate from scratch and save to file
            log.info("Generating poisoned dataset . . .")
            x_poison, y_poison = super().poison(x_trigger, y_trigger, x_train, y_train)
            poison_index = np.where(
                [np.any(p != b) for (p, b) in zip(x_poison, x_train)]
            )[0]

            if filepath is not None:
                # save metadata for verification when dataset is loaded again
                np.savez(
                    filepath,
                    x_poison=x_poison,
                    y_poison=y_poison,
                    poison_index=poison_index,
                    trigger_index=trigger_index,
                    source_class=self.source_class,
                    target_class=self.target_class,
                    percent_poison=self.percent_poison,
                    epsilon=self.epsilon,
                )
            log.info(f"Poisoned dataset saved to {filepath}")

        return x_poison, y_poison, poison_index
