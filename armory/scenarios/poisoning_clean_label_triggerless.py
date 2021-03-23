"""
Classifier evaluation within ARMORY

Scenario Contributor: MITRE Corporation
"""

import logging
from typing import Optional, Tuple
from copy import deepcopy
from collections import defaultdict

import numpy as np
from armory.utils.set_seeds import set_seeds
from tqdm import tqdm
from PIL import ImageOps, Image

from armory.utils.config_loading import (
    load_dataset,
    # load_model,
    load,
    # load_fn,
)
from armory.utils.poison_utils import load_in_memory
from armory.utils import metrics
from armory.scenarios.base import Scenario

logger = logging.getLogger(__name__)


def poison_scenario_preprocessing(batch):
    img_size = 48
    img_out = []
    quantization = 255.0
    for im in batch:
        img_eq = ImageOps.equalize(Image.fromarray(im))
        width, height = img_eq.size
        min_side = min(img_eq.size)
        center = width // 2, height // 2

        left = center[0] - min_side // 2
        top = center[1] - min_side // 2
        right = center[0] + min_side // 2
        bottom = center[1] + min_side // 2

        img_eq = img_eq.crop((left, top, right, bottom))
        img_eq = np.array(img_eq.resize([img_size, img_size])) / quantization

        img_out.append(img_eq)

    return np.array(img_out, dtype=np.float32)


def split_train_target(
    dataset: Tuple[np.ndarray, np.ndarray], n_targets: int, target_class: int
):
    # Given a dataset of form (xs,ys), split it into training and potential targets. Training data
    # consists of all but the last n_target of the datapoints for each class (if there are n_targets
    # or fewer points, it consists of all data and the target is considered invalid). The
    # nontraining points of the chosen target class are returned as potential targets
    target_class = int(target_class)

    total_count_by_class = defaultdict(int)
    curr_count_by_class = defaultdict(int)
    xs, ys = dataset
    for y in ys:
        total_count_by_class[y] += 1

    if total_count_by_class[target_class] <= n_targets:
        raise ValueError(
            f"target_class {target_class} is not a valid target class - fewer "
            f"than {n_targets} data points present"
        )

    xs_train, ys_train, valid_targets = [], [], []
    for idx in range(xs.shape[0]):
        x, y = xs[idx], ys[idx]
        if curr_count_by_class[y] < total_count_by_class[y] - n_targets:
            xs_train.append(x)
            ys_train.append(y)
            curr_count_by_class[y] += 1
        elif y == target_class:
            valid_targets.append(x)

    xs_train = np.array(xs_train)
    ys_train = np.array(ys_train)
    valid_targets = np.array(valid_targets)

    return (xs_train, ys_train), valid_targets


def select_poison_indices(
    n_poison: int,
    clean_train_data: Tuple[np.ndarray, np.ndarray],
    poison_images_class: int,
):
    xs, ys = clean_train_data
    total_target_class = ys[ys == poison_images_class].shape[0]
    if n_poison > total_target_class:
        raise ValueError(
            f"target_class {poison_images_class} is not a valid target class for "
            f"{n_poison} poisons - not enough data points present."
        )
    indices = np.where(ys == poison_images_class)[0][:n_poison]
    return indices


class GTSRB_CLT(Scenario):
    def _evaluate(
        self,
        config: dict,
        num_eval_batches: Optional[int],
        skip_benign: Optional[bool],
        skip_attack: Optional[bool],
        skip_misclassified: Optional[bool],
    ) -> dict:
        config_scenario = config.get("scenario") or {}
        num_runs = config_scenario.get("num_runs", 1)

        results = {}
        for i in range(num_runs):
            _config = deepcopy(config)
            _config["scenario"]["target_idx"] = i
            r = self._evaluate_once(
                _config, num_eval_batches, skip_benign, skip_attack, skip_misclassified
            )
            results[i] = r
        return results

    def _evaluate_once(
        self,
        config: dict,
        num_eval_batches: Optional[int],
        skip_benign: Optional[bool],
        skip_attack: Optional[bool],
        skip_misclassified: Optional[bool],
    ) -> dict:
        """
        Evaluate a config file for classification robustness against attack.

        Note: num_eval_batches shouldn't be set for poisoning scenario and will raise an
        error if it is
        """
        if num_eval_batches:
            raise ValueError("num_eval_batches shouldn't be set for poisoning scenario")
        if skip_benign:
            raise ValueError("skip_benign shouldn't be set for poisoning scenario")
        if skip_attack:
            raise ValueError("skip_attack shouldn't be set for poisoning scenario")
        if skip_misclassified:
            raise ValueError(
                "skip_misclassified shouldn't be set for poisoning scenario"
            )

        set_seeds(config["sysconfig"], config["scenario"])

        classifier_not_art = load(config["model"])  # get_defended_model({}, {})
        classifier_not_art2 = load(config["model"])
        fit_kwargs = config["model"].get("fit_kwargs")

        benign_validation_metric = metrics.MetricList("categorical_accuracy")
        clean_train_data = load_dataset(
            config["dataset"],
            epochs=1,
            split=config["dataset"].get("train_split", "train"),
            preprocessing_fn=poison_scenario_preprocessing,
            shuffle_files=False,
        )
        attack_config = config["poison_attack"]
        train_no_attack = attack_config is None

        if train_no_attack:
            pass
        else:
            tgt_class = attack_config["target_class"]
            poison_images_class = attack_config["poison_class"]

            clean_train_data, candidate_target_images = split_train_target(
                load_in_memory(clean_train_data), n_targets=50, target_class=tgt_class
            )

            target_image = candidate_target_images[
                config["scenario"]["target_idx"] % len(candidate_target_images)
            ]
            # Select, either the first or a random image from the valid targets class
            # target = find_target(clean_train_data, select_first=True)
            poison_gen_classifier, filtering_report = classifier_not_art.defended_train(
                clean_train_data, None, fit_kwargs
            )
            attack_config["args"] = (poison_gen_classifier, target_image)
            attack = load(attack_config)

            n_poison = attack_config["n_poison"]
            poison_base_indices = select_poison_indices(
                n_poison, clean_train_data, poison_images_class
            )
            poison, p_labels = attack.poison(
                clean_train_data[0][poison_base_indices],
                clean_train_data[1][poison_base_indices],
            )
            clean_train_data[0][poison_base_indices] = poison

            classifier, filtering_report = classifier_not_art2.defended_train(
                clean_train_data, None, fit_kwargs
            )

        logger.info("Validating on clean test data")
        test_data = load_dataset(
            config["dataset"],
            epochs=1,
            split=config["dataset"].get("eval_split", "test"),
            preprocessing_fn=poison_scenario_preprocessing,
            shuffle_files=False,
        )
        benign_validation_metric = metrics.MetricList("categorical_accuracy")
        for x, y in tqdm(test_data, desc="Testing"):
            # Ensure that input sample isn't overwritten by classifier
            x.flags.writeable = False
            y_pred = classifier.predict(x)
            benign_validation_metric.append(y, y_pred)
        logger.info(
            f"Unpoisoned validation accuracy: {benign_validation_metric.mean():.2%}"
        )

        results = {
            "benign_validation_accuracy": benign_validation_metric.mean(),
        }

        target_np = np.array([target_image])
        # Ensure that input sample isn't overwritten by classifier
        target_np.flags.writeable = False
        y_pred = classifier.predict(target_np)
        y_pred_lbl = np.argmax(y_pred)
        attack_success = y_pred_lbl == poison_images_class

        logger.info(
            f"Targeted image had label {y_pred_lbl} with true label {tgt_class} "
            f"and the attacker's goal label of {poison_images_class}"
        )
        logger.info(f"Attack success: {attack_success}")

        results["attack_success"] = bool(attack_success)
        results["target_pred_label"] = int(y_pred_lbl)

        return results
