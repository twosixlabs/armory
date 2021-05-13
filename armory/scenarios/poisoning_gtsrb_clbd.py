"""
Classifier evaluation within ARMORY

Scenario Contributor: MITRE Corporation
"""

import logging
from typing import Optional
import os
import random
from copy import deepcopy

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

from art.defences.trainer import AdversarialTrainerMadryPGD

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
            poison_x.append(p_img)
            poison_y.append(p_label)
        else:
            poison_x.append(src_imgs[idx])
            poison_y.append(src_lbls[idx])
    poison_x, poison_y = np.array(poison_x), np.array(poison_y)

    return poison_x, poison_y


class GTSRB_CLBD(Scenario):
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
        proxy_classifier, _ = load_model(model_config)

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
        # Flag for whether to poison dataset -- used to evaluate
        #     performance of defense on clean data
        poison_dataset_flag = config["adhoc"]["poison_dataset"]
        # detect_poison does not currently support data generators
        #     therefore, make in memory dataset
        x_train_all, y_train_all = [], []

        logger.info("Building in-memory dataset for poisoning detection and training")
        for x_train, y_train in clean_data:
            x_train_all.append(x_train)
            y_train_all.append(y_train)
        x_train_all = np.concatenate(x_train_all, axis=0)
        y_train_all = np.concatenate(y_train_all, axis=0)

        if poison_dataset_flag:
            y_train_all_categorical = to_categorical(y_train_all)
            attack_train_epochs = train_epochs
            attack_config = deepcopy(config["attack"])
            use_adversarial_trainer_flag = attack_config.get(
                "use_adversarial_trainer", False
            )

            proxy_classifier_fit_kwargs = {
                "batch_size": fit_batch_size,
                "nb_epochs": attack_train_epochs,
            }
            logger.info("Fitting proxy classifier...")
            if use_adversarial_trainer_flag:
                logger.info("Using adversarial trainer...")
                adversarial_trainer_kwargs = attack_config.pop(
                    "adversarial_trainer_kwargs", {}
                )
                for k, v in proxy_classifier_fit_kwargs.items():
                    adversarial_trainer_kwargs[k] = v
                proxy_classifier = AdversarialTrainerMadryPGD(
                    proxy_classifier, **adversarial_trainer_kwargs
                )
                proxy_classifier.fit(x_train_all, y_train_all)
                attack_config["kwargs"][
                    "proxy_classifier"
                ] = proxy_classifier.get_classifier()
            else:
                proxy_classifier_fit_kwargs["verbose"] = False
                proxy_classifier_fit_kwargs["shuffle"] = True
                proxy_classifier.fit(
                    x_train_all, y_train_all, **proxy_classifier_fit_kwargs
                )
                attack_config["kwargs"]["proxy_classifier"] = proxy_classifier

            attack, backdoor = load(attack_config)

            x_train_all, y_train_all_categorical = attack.poison(
                x_train_all, y_train_all_categorical
            )
            y_train_all = np.argmax(y_train_all_categorical, axis=1)

        if use_poison_filtering_defense:
            y_train_defense = to_categorical(y_train_all)

            defense_config = config["defense"]
            detection_kwargs = config_adhoc.get("detection_kwargs", dict())

            defense_model_config = config_adhoc.get("defense_model", model_config)

            # Assumes classifier_for_defense and classifier use same preprocessing function
            classifier_for_defense, _ = load_model(defense_model_config)
            # ART/Armory API requires that classifier_for_defense trains inside defense_fn
            defense_fn = load_fn(defense_config)
            defense = defense_fn(classifier_for_defense, x_train_all, y_train_defense)

            _, is_clean = defense.detect_poison(**detection_kwargs)
            is_clean = np.array(is_clean)
            logger.info(f"Total clean data points: {np.sum(is_clean)}")

            logger.info("Filtering out detected poisoned samples")
            indices_to_keep = is_clean == 1
            x_train_final = x_train_all[indices_to_keep]
            y_train_final = y_train_all[indices_to_keep]
        else:
            logger.info(
                "Defense does not require filtering. Model fitting will use all data."
            )
            x_train_final = x_train_all
            y_train_final = y_train_all
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
        target_class_benign_metric = metrics.MetricList("categorical_accuracy")
        for x, y in tqdm(test_data, desc="Testing"):
            # Ensure that input sample isn't overwritten by classifier
            x.flags.writeable = False
            y_pred = classifier.predict(x)
            benign_validation_metric.add_results(y, y_pred)
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
            "benign_validation_accuracy": benign_validation_metric.mean(),
            "benign_validation_accuracy_targeted_class": target_class_benign_metric.mean(),
        }

        poisoned_test_metric = metrics.MetricList("categorical_accuracy")
        poisoned_targeted_test_metric = metrics.MetricList("categorical_accuracy")

        if poison_dataset_flag:
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
                    backdoor,
                    poisoned_indices,
                )
                y_pred = classifier.predict(x_test)
                poisoned_test_metric.add_results(y_test, y_pred)

                y_pred_targeted = y_pred[y_test == src_class]
                if len(y_pred_targeted):
                    poisoned_targeted_test_metric.add_results(
                        [tgt_class] * len(y_pred_targeted), y_pred_targeted
                    )
            results["poisoned_test_accuracy"] = poisoned_test_metric.mean()
            results[
                "poisoned_targeted_misclassification_accuracy"
            ] = poisoned_targeted_test_metric.mean()
            logger.info(f"Test accuracy: {poisoned_test_metric.mean():.2%}")
            logger.info(
                f"Test targeted misclassification accuracy: {poisoned_targeted_test_metric.mean():.2%}"
            )

        return results
