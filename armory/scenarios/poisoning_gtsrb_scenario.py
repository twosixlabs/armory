"""
Classifier evaluation within ARMORY

Scenario Contributor: MITRE Corporation
"""

import logging
from typing import Optional
import os
import random

import numpy as np

try:
    from tensorflow import set_random_seed, ConfigProto, Session
    from tensorflow.keras.backend import set_session
    from tensorflow.keras.utils import to_categorical
except ImportError:
    from tensorflow.compat.v1 import (
        set_random_seed,
        ConfigProto,
        Session,
        disable_v2_behavior,
    )
    from tensorflow.compat.v1.keras.backend import set_session
    from tensorflow.compat.v1.keras.utils import to_categorical

    disable_v2_behavior()
from tqdm import tqdm
from PIL import ImageOps, Image

from armory.utils.config_loading import (
    load_dataset,
    load_model,
    load,
    load_fn,
)
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


def poison_dataset(src_imgs, src_lbls, src, tgt, ds_size, attack, poisoned_indices):
    # In this example, all images of "src" class have a trigger
    # added and re-labeled as "tgt" class
    poison_x = []
    poison_y = []
    for idx in range(ds_size):
        if src_lbls[idx] == src and idx in poisoned_indices:
            src_img = src_imgs[idx]
            p_img, p_label = attack.poison(src_img, [tgt])
            poison_x.append(p_img.astype(np.float32))
            poison_y.append(p_label)
        else:
            poison_x.append(src_imgs[idx])
            poison_y.append(src_lbls[idx])

    poison_x, poison_y = np.array(poison_x), np.array(poison_y)

    return poison_x, poison_y


class GTSRB(Scenario):
    def _evaluate(
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
        if config["sysconfig"].get("use_gpu"):
            os.environ["TF_CUDNN_DETERMINISM"] = "1"
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

        model_config = config["model"]
        # Scenario assumes canonical preprocessing_fn is used makes images all same size
        classifier, _ = load_model(model_config)

        config_adhoc = config.get("adhoc") or {}
        train_epochs = config_adhoc["train_epochs"]
        src_class = config_adhoc["source_class"]
        tgt_class = config_adhoc["target_class"]
        fit_batch_size = config_adhoc.get(
            "fit_batch_size", config["dataset"]["batch_size"]
        )

        if not config["sysconfig"].get("use_gpu"):
            conf = ConfigProto(intra_op_parallelism_threads=1)
            set_session(Session(config=conf))

        # Set random seed due to large variance in attack and defense success
        np.random.seed(config_adhoc["split_id"])
        set_random_seed(config_adhoc["split_id"])
        random.seed(config_adhoc["split_id"])
        use_poison_filtering_defense = config_adhoc.get(
            "use_poison_filtering_defense", True
        )
        if self.check_run:
            # filtering defense requires more than a single batch to run properly
            use_poison_filtering_defense = False

        logger.info(f"Loading dataset {config['dataset']['name']}...")

        clean_data = load_dataset(
            config["dataset"],
            epochs=1,
            split=config["dataset"].get("train_split", "train"),
            preprocessing_fn=poison_scenario_preprocessing,
            shuffle_files=False,
        )

        attack_config = config["attack"]
        attack_type = attack_config.get("type")

        fraction_poisoned = config["adhoc"]["fraction_poisoned"]
        # Flag for whether to poison dataset -- used to evaluate
        #     performance of defense on clean data
        poison_dataset_flag = config["adhoc"]["poison_dataset"]
        # detect_poison does not currently support data generators
        #     therefore, make in memory dataset
        x_train_all, y_train_all = [], []

        if attack_type == "preloaded":
            # Number of datapoints in train split of target clasc
            num_images_tgt_class = config_adhoc["num_images_target_class"]
            logger.info(
                f"Loading poison dataset {config_adhoc['poison_samples']['name']}..."
            )
            num_poisoned = int(config_adhoc["fraction_poisoned"] * num_images_tgt_class)
            if num_poisoned == 0:
                raise ValueError(
                    "For the preloaded attack, fraction_poisoned must be set so that at least on data point is poisoned."
                )
            # Set batch size to number of poisons -- read only one batch of preloaded poisons
            config_adhoc["poison_samples"]["batch_size"] = num_poisoned
            poison_data = load_dataset(
                config["adhoc"]["poison_samples"],
                epochs=1,
                split="poison",
                preprocessing_fn=None,
            )

            logger.info(
                "Building in-memory dataset for poisoning detection and training"
            )
            for x_clean, y_clean in clean_data:
                x_train_all.append(x_clean)
                y_train_all.append(y_clean)
            poison_begin = len(np.concatenate(y_train_all, axis=0))
            x_poison, y_poison = poison_data.get_batch()
            x_poison = np.array([xp for xp in x_poison], dtype=np.float32)
            x_train_all.append(x_poison)
            y_train_all.append(y_poison)
            x_train_all = np.concatenate(x_train_all, axis=0)
            y_train_all = np.concatenate(y_train_all, axis=0)
            poison_end = len(y_train_all)
            poisoned_indices = list(range(poison_begin, poison_end))
        else:
            attack = load(attack_config)
            logger.info(
                "Building in-memory dataset for poisoning detection and training"
            )
            for x_train, y_train in clean_data:
                x_train_all.append(x_train)
                y_train_all.append(y_train)
            x_train_all = np.concatenate(x_train_all, axis=0)
            y_train_all = np.concatenate(y_train_all, axis=0)
            if poison_dataset_flag:
                total_count = np.bincount(y_train_all)[src_class]
                poison_count = int(fraction_poisoned * total_count)
                if poison_count == 0:
                    logger.warning(
                        f"No poisons generated with fraction_poisoned {fraction_poisoned} for class {src_class}."
                    )
                src_indices = np.where(y_train_all == src_class)[0]
                poisoned_indices = np.sort(
                    np.random.choice(src_indices, size=poison_count, replace=False)
                )
                x_train_all, y_train_all = poison_dataset(
                    x_train_all,
                    y_train_all,
                    src_class,
                    tgt_class,
                    y_train_all.shape[0],
                    attack,
                    poisoned_indices,
                )
                poisoned_indices = sorted(list([int(x) for x in poisoned_indices]))

        y_train_all_categorical = to_categorical(y_train_all)

        # Flag to determine whether defense_classifier is trained directly
        #     (default API) or is trained as part of detect_poisons method
        fit_defense_classifier_outside_defense = config_adhoc.get(
            "fit_defense_classifier_outside_defense", True
        )
        # Flag to determine whether defense_classifier uses sparse
        #     or categorical labels
        defense_categorical_labels = config_adhoc.get(
            "defense_categorical_labels", True
        )
        if use_poison_filtering_defense:
            if defense_categorical_labels:
                y_train_defense = y_train_all_categorical
            else:
                y_train_defense = y_train_all

            defense_config = config["defense"]
            detection_kwargs = config_adhoc.get("detection_kwargs", dict())

            defense_model_config = config_adhoc.get("defense_model", model_config)
            defense_train_epochs = config_adhoc.get(
                "defense_train_epochs", train_epochs
            )

            # Assumes classifier_for_defense and classifier use same preprocessing function
            classifier_for_defense, _ = load_model(defense_model_config)
            logger.info(
                f"Fitting model {defense_model_config['module']}.{defense_model_config['name']} "
                f"for defense {defense_config['name']}..."
            )
            if fit_defense_classifier_outside_defense:
                classifier_for_defense.fit(
                    x_train_all,
                    y_train_defense,
                    batch_size=fit_batch_size,
                    nb_epochs=defense_train_epochs,
                    verbose=False,
                    shuffle=True,
                )
            defense_fn = load_fn(defense_config)
            defense = defense_fn(classifier_for_defense, x_train_all, y_train_defense)

            _, is_clean = defense.detect_poison(**detection_kwargs)
            is_clean = np.array(is_clean)
            logger.info(f"Total clean data points: {np.sum(is_clean)}")

            logger.info("Filtering out detected poisoned samples")
            indices_to_keep = is_clean == 1
            x_train_final = x_train_all[indices_to_keep]
            y_train_final = y_train_all_categorical[indices_to_keep]

        else:
            logger.info(
                "Defense does not require filtering. Model fitting will use all data."
            )
            x_train_final = x_train_all
            y_train_final = y_train_all_categorical
        if len(x_train_final):
            logger.info(
                f"Fitting model of {model_config['module']}.{model_config['name']}..."
            )
            classifier.fit(
                x_train_final,
                y_train_final,
                batch_size=fit_batch_size,
                nb_epochs=train_epochs,
                verbose=False,
                shuffle=True,
            )
        else:
            logger.warning("All data points filtered by defense. Skipping training")

        logger.info("Validating on clean test data")
        test_data = load_dataset(
            config["dataset"],
            epochs=1,
            split=config["dataset"].get("eval_split", "test"),
            preprocessing_fn=poison_scenario_preprocessing,
            shuffle_files=False,
        )
        benign_validation_metric = metrics.MetricList("categorical_accuracy")
        benign_abstain = metrics.MetricList("abstains")
        target_class_benign_metric = metrics.MetricList("categorical_accuracy")
        benign_actual = []
        benign_predictions = []
        for x, y in tqdm(test_data, desc="Testing"):
            # Ensure that input sample isn't overwritten by classifier
            x.flags.writeable = False
            y_pred = classifier.predict(x)
            benign_actual.extend(y)
            if y_pred.ndim == 1:
                benign_predictions.extend(y_pred)
            else:
                benign_predictions.extend(y_pred.argmax(axis=1))
            benign_validation_metric.add_results(y, y_pred)
            benign_abstain.add_results(y, y_pred)
            y_pred_tgt_class = y_pred[y == src_class]
            if len(y_pred_tgt_class):
                target_class_benign_metric.add_results(
                    [src_class] * len(y_pred_tgt_class), y_pred_tgt_class
                )
        logger.info(
            f"Unpoisoned validation accuracy: {benign_validation_metric.mean():.2%}"
        )
        logger.info(
            f"Unpoisoned validation accuracy on targeted class: {target_class_benign_metric.mean():.2%}"
        )
        results = {
            "benign_categorical_accuracy": benign_validation_metric.values(),
            "benign_mean_categorical_accuracy": benign_validation_metric.mean(),
            "benign_abstains": benign_abstain.values(),
            "benign_mean_abstains": benign_abstain.mean(),
            "benign_categorical_accuracy_targeted_class": target_class_benign_metric.values(),
            "benign_mean_categorical_accuracy_targeted_class": target_class_benign_metric.mean(),
            "benign_y_true": [int(x) for x in benign_actual],
            "benign_y_pred_class": [int(x) for x in benign_predictions],
            "train_poisoned_indices": poisoned_indices,
        }

        if use_poison_filtering_defense:
            results["train_filtered_indices"] = sorted(
                [int(x) for x in np.where(~indices_to_keep)[0]]
            )
        else:
            results["train_filtered_indices"] = []

        results["train_dataset_size"] = len(y_train_all)

        poisoned_test_metric = metrics.MetricList("categorical_accuracy")
        poisoned_abstain = metrics.MetricList("abstains")
        poisoned_targeted_test_metric = metrics.MetricList("categorical_accuracy")
        adversarial_actual = []
        adversarial_predictions = []

        logger.info("Testing on poisoned test data")
        if attack_type == "preloaded":
            test_data_poison = load_dataset(
                config_adhoc["poison_samples"],
                epochs=1,
                split="poison_test",
                preprocessing_fn=None,
            )
            for x_poison_test, y_poison_test in tqdm(
                test_data_poison, desc="Testing poison"
            ):
                x_poison_test = np.array([xp for xp in x_poison_test], dtype=np.float32)
                y_pred = classifier.predict(x_poison_test)
                y_true = [src_class] * len(y_pred)
                adversarial_actual.extend(y_true)
                if y_pred.ndim == 1:
                    adversarial_predictions.extend(y_pred)
                else:
                    adversarial_predictions.extend(y_pred.argmax(axis=1))

                poisoned_targeted_test_metric.add_results(y_poison_test, y_pred)
                poisoned_test_metric.add_results(y_true, y_pred)
                poisoned_abstain.add_results(y_true, y_pred)

            test_data_clean = load_dataset(
                config["dataset"],
                epochs=1,
                split=config["dataset"].get("eval_split", "test"),
                preprocessing_fn=poison_scenario_preprocessing,
                shuffle_files=False,
            )
            for x_clean_test, y_clean_test in tqdm(
                test_data_clean, desc="Testing clean"
            ):
                x_clean_test = np.array([xp for xp in x_clean_test], dtype=np.float32)
                y_pred = classifier.predict(x_clean_test)
                adversarial_actual.extend(y_true)
                if y_pred.ndim == 1:
                    adversarial_predictions.extend(y_pred)
                else:
                    adversarial_predictions.extend(y_pred.argmax(axis=1))
                poisoned_test_metric.add_results(y_clean_test, y_pred)
                poisoned_abstain.add_results(y_clean_test, y_pred)

        elif poison_dataset_flag:
            logger.info("Testing on poisoned test data")
            test_data = load_dataset(
                config["dataset"],
                epochs=1,
                split=config["dataset"].get("eval_split", "test"),
                preprocessing_fn=poison_scenario_preprocessing,
                shuffle_files=False,
            )
            for x_test, y_test in tqdm(test_data, desc="Testing"):
                src_indices = np.where(y_test == src_class)[0]
                poisoned_indices = src_indices  # Poison entire class
                x_test, _ = poison_dataset(
                    x_test,
                    y_test,
                    src_class,
                    tgt_class,
                    len(y_test),
                    attack,
                    poisoned_indices,
                )
                y_pred = classifier.predict(x_test)
                poisoned_test_metric.add_results(y_test, y_pred)
                poisoned_abstain.add_results(y_test, y_pred)
                adversarial_actual.extend(y_test)
                if y_pred.ndim == 1:
                    adversarial_predictions.extend(y_pred)
                else:
                    adversarial_predictions.extend(y_pred.argmax(axis=1))

                y_pred_targeted = y_pred[y_test == src_class]
                if not len(y_pred_targeted):
                    continue
                poisoned_targeted_test_metric.add_results(
                    [tgt_class] * len(y_pred_targeted), y_pred_targeted
                )

        if poison_dataset_flag or attack_type == "preloaded":
            logger.info(f"Test accuracy: {poisoned_test_metric.mean():.2%}")
            logger.info(
                f"Test targeted misclassification accuracy: {poisoned_targeted_test_metric.mean():.2%}"
            )

            results.update(
                {
                    "poison_categorical_accuracy": poisoned_test_metric.values(),
                    "poison_mean_categorical_accuracy": poisoned_test_metric.mean(),
                    "poison_targeted_categorical_accuracy": poisoned_targeted_test_metric.values(),
                    "poison_mean_targeted_categorical_accuracy": poisoned_targeted_test_metric.mean(),
                    "poison_abstains": poisoned_abstain.values(),
                    "poison_mean_abstains": poisoned_abstain.mean(),
                    "poison_y_true": [int(x) for x in adversarial_actual],
                    "poison_y_pred_class": [int(x) for x in adversarial_predictions],
                }
            )

        return results
