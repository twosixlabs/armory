"""
Classifier evaluation within ARMORY
"""

import collections
import importlib
import json
import logging
import shutil
import sys
import time
from typing import Callable

import numpy as np

from armory.data.data import SUPPORTED_DATASETS
from armory.art_experimental import attacks as attacks_extended
from armory.eval import plot

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


SUPPORTED_NORMS = {"L0", "L1", "L2", "Linf"}
SUPPORTED_GOALS = {"targeted", "untargeted"}


# TODO: Merge this with global parser (Issue #59)
def validate_config(config: dict) -> None:
    """
    Validate config for use in this script.
    """
    for k in (
        "adversarial_budget",
        "adversarial_goal",
        "adversarial_knowledge",
        "data",
        "model_file",
        "model_name",
    ):
        if k not in config:
            raise ValueError(f"{k} missing from config")
    if "epsilon" not in config["adversarial_budget"]:
        raise ValueError('"epsilon" missing from config["adversarial_budget"]')
    if config["adversarial_budget"]["epsilon"] != "all":
        raise NotImplementedError("Use 'all' for epsilon for this script")
    goal = config["adversarial_goal"]
    if goal not in SUPPORTED_GOALS:
        raise ValueError(f"{goal} not in supported goals {SUPPORTED_GOALS}")
    if goal == "targeted":
        raise NotImplementedError("targeted goal not complete")
    if "norm" not in config["adversarial_budget"]:
        raise ValueError(f'{k} missing from config["adversarial_budget"]')
    norms = config["adversarial_budget"]["norm"]
    if not isinstance(norms, list):
        norms = [norms]
    for norm in norms:
        if norm not in SUPPORTED_NORMS:
            raise ValueError(f"norm {norm} not in supported norms {SUPPORTED_NORMS}")
        elif norm == "L0":
            raise NotImplementedError("Norm L0 not tested yet")


def project_to_mnist_input(x: np.ndarray, preprocessing_fn: Callable):
    """
    Map input tensor to original space
    """
    if x.dtype not in (float, np.float32):
        raise ValueError("Projection assumes float input")
    if np.isnan(x).any():
        raise ValueError("Cannot project nan values to int input")

    # clip first to deal with any infinite values that may arise
    x = (x.clip(0, 1) * 255.0).round()
    # skip the actual casting to uint8, as we are going to normalize again to float
    return preprocessing_fn(x)


def _evaluate_classifier(config: dict) -> None:
    """
    Evaluate a config file for classification robustness against attack.
    """
    validate_config(config)

    classifier_module = importlib.import_module(config["model_file"])
    classifier = getattr(classifier_module, config["model_name"])
    preprocessing_fn = getattr(classifier_module, "preprocessing_fn")

    # retrofitted to work with existing code
    x_train, y_train, x_test, y_test = SUPPORTED_DATASETS[config["data"]](
        preprocessing_fn=preprocessing_fn
    )

    classifier.fit(
        x_train, y_train, batch_size=64, nb_epochs=1,
    )

    # TODO: Add subsampling (for quick testing)
    subsample = 100
    x_test = x_test[::subsample]
    y_test = y_test[::subsample]

    # TODO: Add defended model to compare against

    # Evaluate the ART classifier on benign test examples
    y_pred = classifier.predict(x_test)
    benign_accuracy = np.sum(np.argmax(y_pred, axis=1) == y_test) / len(y_test)
    logger.info("Accuracy on benign test examples: {}%".format(benign_accuracy * 100))

    # Generate adversarial test examples
    # knowledge = config["adversarial_knowledge"]
    # TODO: add adversarial knowledge

    budget = config["adversarial_budget"]
    norms = budget["norm"]
    if not isinstance(norms, list):
        norms = [norms]

    results = {}
    # Assume min_value = 0
    max_value = 1.0
    input_dim = np.product(x_test.shape[1:])
    norm_map = {  # from norm name to (fgm_input, max_epsilon)
        "L0": (0, input_dim),
        "L1": (1, input_dim * max_value),
        "L2": (2, np.sqrt(input_dim) * max_value),
        "Linf": (np.inf, max_value),
    }
    epsilon_granularity = 1 / 40
    for norm in norms:
        lp_norm, max_epsilon = norm_map[norm]

        # Currently looking at untargeted attacks,
        #     where adversary accuracy ~ 1 - benign accuracy (except incorrect benign)
        attack = attacks_extended.FGMBinarySearch(
            # attack = attacks.FastGradientMethod(
            classifier=classifier,
            norm=lp_norm,
            eps=max_epsilon,
            eps_step=epsilon_granularity * max_epsilon,
            minimal=True,  # find minimum epsilon
        )
        x_test_adv = attack.generate(x=x_test)

        # Map into the original input space (bound and quantize) and back to float
        # NOTE: this step makes many of the attacks fail
        x_test_adv = project_to_mnist_input(x_test_adv, preprocessing_fn)

        diff = (x_test_adv - x_test).reshape(x_test.shape[0], -1)
        epsilons = np.linalg.norm(diff, ord=lp_norm, axis=1)
        if np.isnan(epsilons).any():
            raise ValueError(f"Epsilons have nan values in norm {norm}")
        min_epsilon = 0
        if (epsilons < min_epsilon).any() or (epsilons > max_epsilon).any():
            raise ValueError(f"Epsilons have values outside bounds in norm {norm}")
        # Truncate all epsilons to within min, max bounds [0, max]
        # epsilons = np.clip(epsilons, min_epsilon, max_epsilon)

        y_pred_adv = classifier.predict(x_test_adv)

        # Ignore benign misclassifications - no perturbation needed
        epsilons[np.argmax(y_pred, axis=1) != y_test] = min_epsilon

        # When all attacks fail, set perturbation to None
        epsilons = epsilons.astype(object)
        epsilons[
            (np.argmax(y_pred_adv, axis=1) == y_test)
            & (np.argmax(y_pred, axis=1) == y_test)
        ] = None

        adv_acc = np.sum(np.argmax(y_pred_adv, axis=1) != y_test) / len(y_test)

        # generate curve
        unique_epsilons, accuracy = roc_epsilon(
            epsilons, min_epsilon=min_epsilon, max_epsilon=max_epsilon
        )

        results[norm] = {
            "epsilons": list(unique_epsilons),
            "metric": "Categorical Accuracy",
            "values": list(accuracy),
        }
        # Evaluate the ART classifier on adversarial test examples
        logger.info(
            f"Finished attacking on norm {norm}. Attack success: {adv_acc * 100}%"
        )

    # TODO: This should be moved to the Export module
    filepath = f"outputs/classifier_extended_{int(time.time())}.json"
    with open(filepath, "w") as f:
        output_dict = {
            "config": config,
            "results": results,
        }
        json.dump(output_dict, f, sort_keys=True, indent=4)
    shutil.copyfile(filepath, "outputs/latest.json")

    logger.info(f"Now plotting results")
    plot.classification(filepath)
    plot.classification("outputs/latest.json")


def roc_epsilon(epsilons, min_epsilon=None, max_epsilon=None):
    if not len(epsilons):
        raise ValueError("epsilons cannot be empty")
    total = len(epsilons)
    epsilons = epsilons[epsilons != np.array(None)].astype(float)
    c = collections.Counter()
    c.update(epsilons)
    unique_epsilons, counts = zip(*sorted(list(c.items())))
    unique_epsilons = list(unique_epsilons)
    ccounts = np.cumsum(counts)
    accuracy = list(1 - (ccounts / total))

    if min_epsilon is not None and min_epsilon != unique_epsilons[0]:
        unique_epsilons.insert(0, min_epsilon)
        accuracy.insert(0, accuracy[0])
    if max_epsilon is not None and max_epsilon != unique_epsilons[-1]:
        unique_epsilons.append(max_epsilon)
        accuracy.append(accuracy[-1])  # don't assume perfect attack success

    return [float(x) for x in unique_epsilons], [float(x) for x in accuracy]


def evaluate_classifier(config_path: str) -> None:
    """
    Evaluate a config file for classification robustness against attack.

    Export values.
    """
    with open(config_path) as fp:
        config = json.load(fp)

    _evaluate_classifier(config)


if __name__ == "__main__":
    config_path = sys.argv[-1]
    evaluate_classifier(config_path)
