"""
Classifier evaluation within ARMORY
"""

import collections
from importlib import import_module
import json
import logging
import sys

import numpy as np
from art.attacks import CarliniL2Method, CarliniLInfMethod

from armory.data import data
from armory.eval import plot


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# TODO: Refactor (Issue #112)
def roc_targeted_epsilon(epsilons, min_epsilon=None, max_epsilon=None):
    if not len(epsilons):
        raise ValueError("epsilons cannot be empty")
    total = len(epsilons)
    epsilons = epsilons[epsilons != np.array(None)].astype(float)
    c = collections.Counter()
    c.update(epsilons)
    unique_epsilons, counts = zip(*sorted(list(c.items())))
    unique_epsilons = list(unique_epsilons)
    ccounts = np.cumsum(counts)
    targeted_attack_success = list((ccounts / total))

    if min_epsilon is not None and min_epsilon != unique_epsilons[0]:
        unique_epsilons.insert(0, min_epsilon)
        targeted_attack_success.insert(0, targeted_attack_success[0])
    if max_epsilon is not None and max_epsilon != unique_epsilons[-1]:
        unique_epsilons.append(max_epsilon)
        targeted_attack_success.append(
            targeted_attack_success[-1]
        )  # don't assume perfect attack success

    return (
        [float(x) for x in unique_epsilons],
        [float(x) for x in targeted_attack_success],
    )


def evaluate_classifier(config_path: str) -> None:
    """
    Evaluate a config file for classiifcation robustness against attack.
    """
    batch_size = 64
    epochs = 3
    num_classes = 10
    num_attacked_pts = 100
    seed = 123

    with open(config_path, "r") as fp:
        config = json.load(fp)

    classifier_module = import_module(config["model_file"])
    classifier = getattr(classifier_module, config["model_name"])
    preprocessing_fn = getattr(classifier_module, "preprocessing_fn")

    x_train, y_train, x_test, y_test = data.load(
        config["data"], preprocessing_fn=preprocessing_fn
    )

    classifier.fit(x_train, y_train, batch_size=batch_size, nb_epochs=epochs)

    # Generate adversarial test examples
    if config["attack"] == "CarliniL2Method":
        attack = CarliniL2Method(classifier=classifier, targeted=True)
        norm = "L2"
        lp_norm = 2
    elif config["attack"] == "CarliniLInfMethod":
        attack = CarliniLInfMethod(classifier, targeted=True)
        norm = "Linf"
        lp_norm = np.inf
    else:
        raise ValueError(
            f"Invalid attack {config['attack']}: only CarliniL2Method and CarliniInfMethod supported"
        )

    y_target = (y_test + 1) % num_classes

    np.random.seed(seed)
    indices = np.random.choice(x_test.shape[0], num_attacked_pts)

    x_test_sample = x_test[indices]
    y_test_sample = y_test[indices]
    y_target_sample = y_target[indices]

    x_test_adv = attack.generate(x=x_test_sample, y=y_target_sample)

    diff = (x_test_adv - x_test_sample).reshape(x_test_adv.shape[0], -1)
    epsilons = np.linalg.norm(diff, ord=lp_norm, axis=1)

    y_clean_pred = np.argmax(classifier.predict(x_test_sample), axis=1)
    y_adv_pred = np.argmax(classifier.predict(x_test_adv), axis=1)

    # Evaluate the ART classifier on adversarial test examples and clean test examples
    successful_attack_indices = (y_clean_pred != y_target_sample) & (
        y_adv_pred == y_target_sample
    )

    benign_misclassification_rate = np.sum(y_clean_pred == y_target_sample) / float(
        y_clean_pred.shape[0]
    )

    logger.info(
        f"Benign misclassification as targeted examples: {benign_misclassification_rate * 100}%"
    )

    targeted_attack_success_rate = np.sum(successful_attack_indices) / float(
        y_clean_pred.shape[0]
    )
    clean_accuracy = np.sum(y_clean_pred == y_test_sample) / float(
        y_clean_pred.shape[0]
    )

    logger.info(f"Accuracy on benign test examples: {clean_accuracy * 100}%")

    epsilons = epsilons.astype(object)
    epsilons[np.logical_not(successful_attack_indices)] = None

    unique_epsilons, targeted_attack_success = roc_targeted_epsilon(epsilons)
    results = {}

    results[norm] = {
        "epsilons": list(unique_epsilons),
        "metric": "Targeted attack success rate",
        "values": list(targeted_attack_success),
    }

    logger.info(
        f"Finished attacking on norm {norm}. Attack success: {targeted_attack_success_rate * 100}%"
    )

    filepath = f"outputs/carlini_wagner_attack_{norm}_targeted_output.json"
    with open(filepath, "w") as f:
        output_dict = {
            "config": config,
            "results": results,
        }
        json.dump(output_dict, f, sort_keys=True, indent=4)
    plot.classification(filepath)


if __name__ == "__main__":
    config_path = sys.argv[-1]
    evaluate_classifier(config_path)
