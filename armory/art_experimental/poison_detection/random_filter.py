import numpy as np

from art.defences.detector.poison import PoisonFilteringDefence

from armory.logs import log


class RandomFilterBaselineDefense(PoisonFilteringDefence):


    def __init__(self, classifier, x_train, y_train, **kwargs):
        """
        Create a :class:`.RandomFilterBaselineDefense` object with the provided classifier.
        :param classifier: Model evaluated for poison.
        :param x_train: dataset used to train the classifier.
        :param y_train: labels used to train the classifier.
        """
        super().__init__(classifier, x_train, y_train)
        self.n_data = len(y_train)

    def evaluate_defence(self, is_clean, **kwargs):
        raise NotImplementedError("evaluate_defence() not implemented for RandomFilterBaselineDefense")


    def detect_poison(self, **kwargs):
        """
        Selects data at random to label as poison.
        :return: (report, is_clean_lst):
                where report is None (for future ART compatibility)
                where is_clean is a list, where a 1 at index i indicates that x_train[i] is clean,
                    and a 0 indicates that x_train[i] is detected as poison.
        """
        
        fraction_to_remove = kwargs.get("fraction_to_remove", None)
        if fraction_to_remove is None:
            fraction_to_remove = 0.1
            log.info("Setting fraction_to_remove to 0.1.  This can be set under defense/kwargs in the config")
        
        if fraction_to_remove < 0 or fraction_to_remove > 1:
            raise ValueError(f"defense/kwargs/fraction_to_remove must be set between 0 and 1 in the config.  Got {fraction_to_remove}")


        is_clean = np.random.choice([0,1], self.n_data, p=[fraction_to_remove, 1 - fraction_to_remove])

        return None, is_clean