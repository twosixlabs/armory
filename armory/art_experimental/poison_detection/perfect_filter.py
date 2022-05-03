
from art.defences.detector.poison import PoisonFilteringDefence

"""
NOTE: PERFECT FILTERING REQUIRES KNOWLEDGE OF THE POISON INDEX,
WHICH THIS MODULE CANNOT ACQUIRE.

PERFECT FILTERING IS THEREFORE IMPLEMENTED IN SCENARIO CODE,
AND WILL OCCUR UPON THE LOADING OF THIS MODULE
ALTHOUGH NO FUNCTIONS HEREIN WILL BE CALLED.

THE ONLY POINT OF THIS CLASS IS TO SATISFY THE CONFIG VALIDATION REQUIREMENTS.

THIS IS WEIRD, BUT AT LEAST THE CONFIG INTERFACE IS THE SAME AS FOR OTHER DEFENSES.

"""


class PerfectFilterBaselineDefense(PoisonFilteringDefence):
    def __init__(self, classifier, x_train, y_train, **kwargs):
        """
        Create a :class:`.PerfectFilterBaselineDefense` object with the provided classifier.

        :param classifier: Model evaluated for poison.
        :param x_train: dataset used to train the classifier.
        :param y_train: labels used to train the classifier.
        """
        super().__init__(classifier, x_train, y_train)

    def evaluate_defence(self, is_clean, **kwargs):
        raise NotImplementedError(
            "evaluate_defence() not implemented for PerfectFilterBaselineDefense"
        )

    def detect_poison(self, **kwargs):
        """
        Raises an error explaining that the perfect_filter doesn't actually use this function.

        """
        raise NotImplementedError(
            "detect_poison() not implemented for PerfectFilterBaselineDefense. "
            + "This is because a filter module that is distinct from the scenario cannot know which samples were poisoned. "
            + "The scenario code will filter all poison automatically if this defense is requested."
        )
