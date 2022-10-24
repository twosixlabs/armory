from art.defences.detector.poison import PoisonFilteringDefence
import numpy as np

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
        raise NotImplementedError(
            "evaluate_defence() not implemented for RandomFilterBaselineDefense"
        )

    def detect_poison(self, expected_pp_poison=None):
        """
        Selects data at random to label as poison.
        :return: (report, is_clean_lst):
                where report is None (for future ART compatibility)
                where is_clean is a list, where a 1 at index i indicates that x_train[i] is clean,
                    and a 0 indicates that x_train[i] is detected as poison.
        """

        if expected_pp_poison is None:
            expected_pp_poison = 0.3
            log.info(
                "Setting expected_pp_poison to 0.3.  This can be set under defense/kwargs in the config"
            )

        if expected_pp_poison < 0 or expected_pp_poison > 1:
            raise ValueError(
                f"defense/kwargs/expected_pp_poison must be set between 0 and 1 in the config.  Got {expected_pp_poison}"
            )

        is_clean = np.random.choice(
            [0, 1], self.n_data, p=[expected_pp_poison, 1 - expected_pp_poison]
        )

        return None, is_clean
