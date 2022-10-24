import os

from art.attacks.poisoning.gradient_matching_attack import GradientMatchingAttack
import numpy as np

from armory.logs import log


class GradientMatchingWrapper(GradientMatchingAttack):
    def __init__(self, classifier, **kwargs):
        self.attack_kwargs = kwargs
        self.source_class = kwargs.pop("source_class")
        self.target_class = kwargs.pop("target_class")
        self.triggers_chosen_randomly = kwargs.pop("triggers_chosen_randomly")
        learning_rate_schedule = tuple(kwargs.pop("learning_rate_schedule"))

        self.overwrite_presaved_data = (
            False
            if "overwrite_presaved_data" not in kwargs
            else kwargs.pop("overwrite_presaved_data")
        )

        super().__init__(
            classifier=classifier,
            learning_rate_schedule=learning_rate_schedule,
            **kwargs,
        )

    def _check_for_config_conflicts(
        self,
        filepath,
        trigger_index,
        load_trigger_index,
        load_source_class,
        load_target_class,
        percent_poison,
        epsilon,
    ):
        # Check that config parameters are consistent with pre-poisoned dataset and raise a super helpful error if not
        # If config is consistent with pre-saved data, do nothing.

        loaded_description = f"Adversarial dataset at filepath {filepath} is not consistent with config.  The saved dataset was trained:\n"
        config_description = "But the config is requesting:\n"
        suggestion = (
            "Please clarify your intent by doing one of the following:\n"
            + "  1 Reconcile the config with the above parameters (Armory can then use the saved dataset)--see note below,\n"
            + "  2 Delete or rename the saved dataset (Armory will generate a new dataset with the old filename),\n"
            + "  3 Change attack/kwargs/data_filepath in the config (Armory will generate a new dataset with a new filename),\n"
            + "  4 Set attack/kwargs/overwrite_presaved_data to True in the config (Armory will generate a new dataset and overwrite the saved one).\n"
            + "Note for 1: Either source_class or trigger_index can be set to null, or all three of source_class, trigger_index, and target_class.\n"
        )

        config_conflict = False

        if self.triggers_chosen_randomly:
            # If random_triggers is True, we can override them with loaded data because user didn't care exactly which trigger images.
            # But, we still enforce that there are the correct _nummber_ of triggers.
            if len(load_trigger_index) != len(trigger_index):
                loaded_description += (
                    f"-  for {len(load_trigger_index)} trigger image(s)\n"
                )
                config_description += f"-  {len(trigger_index)} trigger image(s)\n"
                config_conflict = True
        else:
            # if random_triggers is false, trigger_index must match exactly.
            if sorted(load_trigger_index) != sorted(trigger_index):
                loaded_description += f"-  for trigger image(s) {load_trigger_index}\n"
                config_description += f"-  trigger image(s) {trigger_index}\n"
                config_conflict = True

        # source class must match element-wise
        if not (
            len(load_source_class) == len(self.target_class)
            and sum([s1 == s2 for s1, s2 in zip(load_source_class, self.source_class)])
            == len(load_source_class)
        ):
            loaded_description += f"-  for source class(es) {load_source_class}\n"
            config_description += f"-  source class(es) {self.source_class}\n"
            config_conflict = True

        # target class must match element-wise
        if not (
            len(load_target_class) == len(self.target_class)
            and sum([t1 == t2 for t1, t2 in zip(load_target_class, self.target_class)])
            == len(load_target_class)
        ):
            loaded_description += f"-  for target class(es) {load_target_class}\n"
            config_description += f"-  target class(es) {self.target_class}\n"
            config_conflict = True

        # Fraction poisoned and epsilon must also match
        if percent_poison != self.percent_poison:
            loaded_description += f"-  with {percent_poison} fraction poisoned\n"
            config_description += f"-  {self.percent_poison} fraction poisoned\n"
            config_conflict = True
        if epsilon != self.epsilon:
            loaded_description += f"-  with an L_inf bound of {epsilon}\n"
            config_description += f"-  an L_inf bound of {self.epsilon}\n"
            config_conflict = True

        if config_conflict:
            raise ValueError(loaded_description + config_description + suggestion)

    def poison(self, filepath, x_trigger, y_trigger, x_train, y_train, trigger_index):
        """
        Return a dataset poisoned to cause misclassification of x_trigger.
        If "filepath" already exists, will attempt to load pre-poisoned dataset from that location.
        Otherwise, generates poisoned dataset and saves to "filepath".
        If "filepath" is None, generates dataset but does not save it.

        filepath, string: path to load data from (if it exists) or save it to (if it doesn't)
        x_trigger, array: Images from the test set that we hope to misclassify
        y_trigger, array: Target labels for misclassification
        x_train, array: The training images to be poisoned
        y_train, array: Labels for x_train
        trigger_index, int or array: Index or indices of trigger images
        """

        if self.overwrite_presaved_data:
            if os.path.exists(filepath):
                log.info("Deleting pre-saved dataset and generating anew.")
                os.remove(filepath)

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
            load_target_class, load_source_class, percent_poison, epsilon = (
                poison_npz["target_class"],
                poison_npz["source_class"],
                poison_npz["percent_poison"],
                poison_npz["epsilon"],
            )

            if len(x_trigger) == 0 and len(y_trigger) == 0:
                # Config didn't give attack parameters so we can just return the loaded data
                return (
                    x_poison,
                    y_poison,
                    poison_index,
                    load_trigger_index,
                    load_source_class,
                    load_target_class,
                )

            self._check_for_config_conflicts(
                filepath,
                trigger_index,
                load_trigger_index,
                load_source_class,
                load_target_class,
                percent_poison,
                epsilon,
            )

            trigger_index = list(load_trigger_index)
            source_class = list(load_source_class)
            target_class = list(load_target_class)

        else:
            if len(x_trigger) == 0 and len(y_trigger) == 0:
                # Config didn't give attack parameters but there was no saved dataset
                raise ValueError(
                    "Config must contain either a filepath to an existing presaved dataset, or values for trigger_index, source_class, and target_class"
                )

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
            else:
                log.warning(
                    "If you wish the poisoned dataset to be saved, please set attack/kwargs/data_filepath in the config."
                )
            source_class = self.source_class
            target_class = self.target_class

        # Return source, target, and trigger in case they were None/empty and modified by this function
        return (
            x_poison,
            y_poison,
            poison_index,
            trigger_index,
            source_class,
            target_class,
        )
