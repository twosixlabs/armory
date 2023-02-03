"""
Functions for manipulating outputs from scenario evaluations

NOTE: to plot, "record_metric_per_sample" in config must be True

Example:
```
filepath = "<output_dir>/scenario_results.json"
results = outputs.Results.from_file(filepath, name="Test")
results.plot()  # plot and show
results.plot("results_figure.png")  # save results as png
```

"""

import collections
import json

import numpy as np


def to_list(element):
    if isinstance(element, list):
        return element
    elif element is None:
        return []
    elif isinstance(element, str):
        return [element]
    else:  # attempt to cast to list (e.g., for tuples, iterables)
        return list(element)


class Results:
    def __init__(self, json_dict, name=None):
        if name is None:
            name = "Unnamed"
        self.name = name

        try:
            self.metric = json_dict["config"]["metric"]
            self.perturbation_metrics = to_list(self.metric["perturbation"])
            self.tasks = to_list(self.metric["task"])
            self.results = json_dict["results"]
            self.means = self.metric["means"]
            self.record_metric_per_sample = self.metric["record_metric_per_sample"]
        except KeyError as e:
            raise KeyError(f"{str(e)} not found in json dictionary")

        if len(self.perturbation_metrics) != 1:
            raise NotImplementedError("number of perturbation metrics must be 1")
        if len(self.tasks) != 1:
            raise NotImplementedError("number of task metrics must be 1")
        self.task = self.tasks[0]
        self.perturbation_metric = self.perturbation_metrics[0]

        try:
            if self.means:
                self.mean_benign_task = self.results[f"benign_mean_{self.task}"]
                self.mean_adversarial_task = self.results[
                    f"adversarial_mean_{self.task}"
                ]
                self.mean_targeted_task = self.results.get(
                    f"targeted_mean_{self.task}", None
                )
                self.mean_perturbation = self.results[
                    f"perturbation_mean_{self.perturbation_metric}"
                ]
            if self.record_metric_per_sample:
                self.benign_task = self.results[f"benign_{self.task}"]
                self.adversarial_task = self.results[f"adversarial_{self.task}"]
                self.targeted_task = self.results.get(f"targeted_{self.task}", None)
                self.perturbation = self.results[
                    f"perturbation_{self.perturbation_metric}"
                ]
        except KeyError as e:
            raise KeyError(f"{str(e)} not found in results portion of json dictionary")

    @classmethod
    def from_file(cls, filepath, name=None):
        with open(filepath) as f:
            json_dict = json.load(f)
        return cls(json_dict, name=name)

    def perturbation_accuracy(self, targeted):
        if targeted:
            return perturbation_accuracy(
                self.benign_task, self.targeted_task, self.perturbation
            )
        else:
            return perturbation_accuracy(
                self.benign_task, self.adversarial_task, self.perturbation
            )

    def plot(self, filepath=None, max_epsilon=None, targeted=False):
        """
        Plot example. If filepath is None, show it, else save to filepath
        """
        epsilons, acc = self.perturbation_accuracy(targeted=targeted)
        if max_epsilon is None:
            max_epsilon = max(epsilons)

        import matplotlib.pyplot as plt

        plt.plot(epsilons, acc, "bx-")
        plt.xlabel(f"{self.perturbation_metric}")
        plt.ylabel(f"{self.task}")
        plt.title(f"{self.name} Perturbation-Accuracy Graph")
        plt.axis([0, max_epsilon, 0, 1])
        if filepath is None:
            plt.show()
        else:
            plt.savefig(filepath)


def perturbation_accuracy(benign, adversarial, perturbations, max_perturbation=None):
    """
    Returns perturbation-accuracy plot points
    """
    benign, adversarial, perturbations = (
        np.asarray(x) for x in (benign, adversarial, perturbations)
    )
    if not (benign.shape == adversarial.shape == perturbations.shape):
        raise ValueError("values are different shape")
    if not len(benign):
        return [], []
    if benign.dtype != int or adversarial.dtype != int:
        raise ValueError("task metrics are not lists of ints")
    if (
        benign.max() > 1
        or adversarial.max() > 1
        or benign.min() < 0
        or adversarial.min() < 0
    ):
        raise ValueError("benign and adversarial inputs must be in (0, 1)")
    benign, adversarial = (x.astype(bool) for x in (benign, adversarial))

    for epsilon in perturbations:
        if epsilon < 0:
            raise ValueError("perturbations cannot be negative")
    if max_perturbation is not None:
        if max_perturbation < max(perturbations):
            raise ValueError(
                f"max_perturbation {max_perturbation} < max(perturbations) {max(perturbations)}"
            )
    c = collections.Counter()
    c[0] = 0  # initialize perfect accuracy on benign
    total = len(benign)
    for b, a, p in zip(benign, adversarial, perturbations):
        if not b:  # misclassification (no perturbation needed)
            c.update([0])
        elif not a:  # successful untargeted adversarial attack
            c.update([p])
        else:  # ignore unsuccessful attacks
            pass

    unique_epsilons, counts = zip(*sorted(list(c.items())))
    unique_epsilons = list(unique_epsilons)
    ccounts = np.cumsum(counts)
    accuracy = list(1 - (ccounts / total))
    return np.array(unique_epsilons), np.array(accuracy)
