"""
Data structures for exporting ARMORY evaluation data.
"""
import json


class Export(object):
    def __init__(self, baseline_accuracy: float, adversarial_accuracy: float):
        self.data = {
            "baseline_accuracy": baseline_accuracy,
            "adversarial_accuracy": adversarial_accuracy,
        }

    def save(self) -> None:
        with open("outputs/evaluation-results.json", "w") as fp:
            json.dump(self.data, fp)
