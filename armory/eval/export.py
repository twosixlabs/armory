"""
Data structures for exporting ARMORY evaluation data.
"""
import json


class Export(object):
    def __init__(
        self, performer: str, baseline_accuracy: str, adversarial_accuracy: str
    ):
        self.data = {
            "performer": performer,
            "baseline_accuracy": baseline_accuracy,
            "adversarial_accuracy": adversarial_accuracy,
        }

    def save(self) -> None:
        with open("outputs/evaluation-results.json", "w") as fp:
            json.dump(self.data, fp)
