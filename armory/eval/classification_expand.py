"""
Classifier evaluation within ARMORY
"""

import collections
import importlib
import json
import logging
import sys

import numpy as np

from armory.eval.export import Export
from armory.webapi.data import SUPPORTED_DATASETS
from armory.art_experimental import attacks as attacks_extended
from art import attacks

log = logging.getLogger(__name__)


def _evaluate_classifier(config: dict) -> None:
    """
    Evaluate a config file for classification robustness against attack.
    """

    # TODO: Preprocessing needs to be refactored elsewhere!
    def _normalize_img(img_batch):
        norm_batch = img_batch.astype(np.float32) / 255.0
        return norm_batch

    classifier_module = importlib.import_module(config["model_file"])
    classifier = getattr(classifier_module, config["model_name"])

    train_ds, test_ds = SUPPORTED_DATASETS[config["data"]]()

    x_train, y_train = train_ds["image"], train_ds["label"]
    x_test, y_test = test_ds["image"], test_ds["label"]
    x_train, x_test = _normalize_img(x_train), _normalize_img(x_test)

    classifier.fit(
        x_train, y_train, batch_size=64, nb_epochs=3,
    )

    # Evaluate the ART classifier on benign test examples
    y_pred = classifier.predict(x_test)
    benign_accuracy = np.sum(np.argmax(y_pred, axis=1) == y_test) / len(y_test)
    log.info("Accuracy on benign test examples: {}%".format(benign_accuracy * 100))

    # Generate adversarial test examples
    knowledge = config["adversarial_knowledge"]
    # TODO: add adversarial knowledge

    budget = config["adversarial_budget"]
    norms = budget["norm"]
    if not isinstance(norms, list):
        norms = [norms]

    for norm in norms:
        if config["adversarial_budget"]["epsilon"] == "all":
            fgm_norm_map = {
                "L1": 1,
                "L2": 2,
                "Linf": np.inf,
            }
            if norm not in fgm_norm_map:
                raise ValueError(f"norm {norm} not valid for fgm")

            attack = attacks_extended.FGMBinarySearch(
            #attack = attacks.FastGradientMethod(
                classifier=classifier,
                norm=fgm_norm_map[norm],
                eps=1.0,
                eps_step=0.1,
                minimal=True,  # find minimum epsilon
            )
        else:
            raise NotImplementedError("Use 'all' for epsilon")
        x_test_adv = attack.generate(x=x_test)

        # TODO: should we quantize to [0, 1, ..., 255] for MNIST?
        diff = (x_test_adv - x_test).reshape(x_test.shape[0], -1)
        epsilons = np.linalg.norm(diff, ord=fgm_norm_map[norm], axis=1)
        y_pred_adv = classifier.predict(x_test_adv)

        # Ignore benign misclassifications - no perturbation needed
        min_value, max_value = 0, 1
        epsilons[np.argmax(y_pred, axis=1) != y_test] = min_value

        # Truncate all epsilons to within min, max bounds [0, max]
        epsilons = np.clip(epsilons, min_value, max_value)

        # When all attacks fail, set perturbation to max
        epsilons[
            (np.argmax(y_pred_adv, axis=1) == y_test)
            & (np.argmax(y_pred, axis=1) == y_test)
        ] = max_value

        # generate curve
        adversarial_accuracy = np.sum(np.argmax(y_pred_adv, axis=1) == y_test) / len(
            y_test
        )

    # failed examples:

    # verify epsilon values?

    # Evaluate the ART classifier on adversarial test examples
    log.info(
        "Accuracy on adversarial test examples: {}%".format(adversarial_accuracy * 100)
    )


def roc_epsilon(epsilons, min_value=None, max_value=None):
    if not len(epsilons):
        raise ValueError("epsilons cannot be empty")
    c = collections.Counter()
    c.update(epsilons)
    unique_epsilons, counts = zip(*sorted(list(c.items())))
    unique_epsilons = np.array(unique_epsilons)
    ccounts = np.cumsum(counts)
    accuracy = list(ccounts / ccounts[-1])
    if min_value is not None and min_value != unique_epsilons[0]:
        unique_epsilons.insert(0, min_value)
        accuracy.insert(0, 0)
    if max_value is not None and max_value != unique_epsilons[-1]:
        unique_epsilons.append(max_value)
        accuracy.append(1)

    return np.array(unique_epsilons), np.array(accuracy)


def evaluate_classifier(config_path: str) -> None:
    """
    Evaluate a config file for classification robustness against attack.

    Export values.
    """
    with open(config_path) as fp:
        config = json.load(fp)

    benign_accuracy, adversarial_accuracy = _evaluate_classifer(config)

    exporter = Export(benign_accuracy, adversarial_accuracy)
    exporter.save()


def set_attack(classifier, method, epsilon):
    pass


# TODO: load training set adversarial examples (for optimal attack line) ??

if __name__ == "__main__":
    config_path = sys.argv[-1]
    evaluate_classifier(config_path)
