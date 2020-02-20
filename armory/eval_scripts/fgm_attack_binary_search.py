"""
Classifier evaluation within ARMORY
"""

import collections
import json
import logging
import os
import shutil
import sys
import time
from importlib import import_module
from typing import Callable

import numpy as np

from armory.utils.config_loading import load_dataset, load_model
from armory.eval import plot
from armory.paths import DockerPaths

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


SUPPORTED_NORMS = {"L0", "L1", "L2", "Linf"}
SUPPORTED_GOALS = {"targeted", "untargeted"}


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


def evaluate_classifier(config_path: str) -> None:
    """
    Evaluate a config file for classification robustness against attack.
    """
    docker_paths = DockerPaths()

    with open(config_path) as fp:
        config = json.load(fp)

    model_config = config["model"]
    classifier, preprocessing_fn = load_model(model_config)

    logger.info(f"Loading dataset {config['dataset']['name']}...")
    x_train, y_train, x_test, y_test = load_dataset(
        config["dataset"], preprocessing_fn=preprocessing_fn
    )

    logger.info(
        f"Fitting clean unpoisoned model of {model_config['module']}.{model_config['name']}..."
    )
    classifier.fit(
        x_train,
        y_train,
        batch_size=config["adhoc"]["batch_size"],
        nb_epochs=config["adhoc"]["epochs"],
    )

    # Speeds up testing...
    subsample = 100
    x_test = x_test[::subsample]
    y_test = y_test[::subsample]

    # Evaluate the ART classifier on benign test examples
    y_pred = classifier.predict(x_test)
    benign_accuracy = np.sum(np.argmax(y_pred, axis=1) == y_test) / len(y_test)
    logger.info("Accuracy on benign test examples: {}%".format(benign_accuracy * 100))

    attack_config = config["attack"]
    attack_module = import_module(attack_config["module"])
    attack_fn = getattr(attack_module, attack_config["name"])
    budget = attack_config["budget"]
    norms = budget["norm"]

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
    for norm in norms:
        lp_norm, max_epsilon = norm_map[norm]

        # Currently looking at untargeted attacks,
        # where adversary accuracy ~ 1 - benign accuracy (except incorrect benign)
        attack = attack_fn(
            classifier=classifier,
            norm=lp_norm,
            eps=max_epsilon,
            **attack_config["kwargs"],
        )
        logger.info(f"Generating adversarial examples for norm {norm}...")
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

    logger.info("Saving json output...")
    filepath = os.path.join(
        docker_paths.output_dir, f"classifier_extended_{int(time.time())}.json"
    )
    with open(filepath, "w") as f:
        output_dict = {
            "config": config,
            "results": results,
        }
        json.dump(output_dict, f, sort_keys=True, indent=4)
    shutil.copyfile(filepath, os.path.join(docker_paths.output_dir, "latest.json"))

    logger.info(f"Now plotting results...")
    plot.classification(filepath)
    plot.classification(os.path.join(docker_paths.output_dir, "latest.json"))


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


if __name__ == "__main__":
    config_path = sys.argv[-1]
    evaluate_classifier(config_path)
