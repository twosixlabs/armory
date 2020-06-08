"""
Classifier evaluation within ARMORY

Scenario Contributor: MITRE Corporation
"""

import logging

import numpy as np
from tensorflow import set_random_seed
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

from armory.utils.config_loading import (
    load_dataset,
    load_model,
    load,
    load_fn,
)
from armory.utils import metrics
from armory.scenarios.base import Scenario

logger = logging.getLogger(__name__)


class GTSRB(Scenario):
    def _evaluate(self, config: dict) -> dict:
        """
        Evaluate a config file for classification robustness against attack.
        """

        model_config = config["model"]
        classifier, preprocessing_fn = load_model(model_config)

        config_adhoc = config.get("adhoc") or {}
        train_epochs = config_adhoc["train_epochs"]
        src_class = config_adhoc["source_class"]
        tgt_class = config_adhoc["target_class"]

        # Set random seed due to large variance in attack and defense success
        np.random.seed(config_adhoc["np_seed"])
        set_random_seed(config_adhoc["tf_seed"])
        use_poison_filtering_defense = config_adhoc.get(
            "use_poison_filtering_defense", True
        )

        logger.info(f"Loading dataset {config['dataset']['name']}...")
        batch_size = config["dataset"]["batch_size"]
        clean_data = load_dataset(
            config["dataset"],
            epochs=1,
            split_type="train",
            preprocessing_fn=preprocessing_fn,
        )

        logger.info(f"Loading poison dataset {config['poison_samples']['name']}...")
        batch_size = config["poison_samples"]["batch_size"]
        poison_data = load_dataset(
            config["poison_samples"],
            epochs=1,
            split_type="poison",
            preprocessing_fn=None,
        )

        logger.info("Building in-memory dataset for poisoning detection and training")
        attack_config = config["attack"]
        attack = load(attack_config)
        poison_dataset_flag = config["adhoc"]["poison_dataset"]
	# Concatenate the clean and poisoned samples
        x_train_all, y_train_all = [], []
        for x_clean, y_clean in clean_data:
            x_poison, y_poison = poison_data.get_batch()
            x_poison = np.array([xp for xp in x_poison], dtype = np.float)
            x_train_all.append(x_clean)
            y_train_all.append(y_clean)
            x_train_all.append(x_poison)
            y_train_all.append(y_poison)
        
        x_train_all = np.concatenate(x_train_all, axis=0)
        y_train_all = np.concatenate(y_train_all, axis=0)

        y_train_all_categorical = to_categorical(y_train_all)

        if use_poison_filtering_defense:
            defense_config = config["defense"]
            logger.info(
                f"Fitting model {model_config['module']}.{model_config['name']} "
                f"for defense {defense_config['name']}..."
            )
            classifier_for_defense, _ = load_model(model_config)
            classifier_for_defense.fit(
                x_train_all,
                y_train_all_categorical,
                batch_size=512,
                nb_epochs=train_epochs,
                verbose=False,
            )
            defense_fn = load_fn(defense_config)
            defense = defense_fn(
                classifier_for_defense, x_train_all, y_train_all_categorical
            )

            _, is_clean = defense.detect_poison(nb_clusters=2, nb_dims=43, reduce="PCA")
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
                batch_size=512,
                nb_epochs=train_epochs,
                verbose=False,
            )
        else:
            logger.warning("All data points filtered by defense. Skipping training")

        logger.info("Validating on poisoned test data")
        test_data = load_dataset(
            config["poison_samples"],
            epochs=1,
            split_type="poison_test",
            preprocessing_fn=None,
        )
        validation_metric = metrics.MetricList("categorical_accuracy")
        x_test_all, y_test_all = [], []
        for x_test, y_test in test_data:
            x_poison_test, y_poison_test = test_data.get_batch()
            x_poison_test = np.array([xp for xp in x_poison_test], dtype = np.float)
            x_test_all.append(x_poison_test)
            y_test_all.append(y_poison_test)

        x_test_all = np.concatenate(x_test_all, axis=0)
        y_test_all = np.concatenate(y_test_all, axis=0)

        fraction_poisoned = config["adhoc"]["fraction_poisoned"]
        num_test_all = x_test_all.shape[0]
        num_test = round(fraction_poisoned * num_test_all)
        x_test = x_test_all[0:num_test,:,:,:]
        y_test = y_test_all[0:num_test]
        y_test_all_categorical = to_categorical(y_test)
        y_pred = np.argmax(classifier.predict(x_test), axis=1)
        validation_metric.append(y_test, y_pred)
        logger.info(f"poisoned validation accuracy: {validation_metric.mean():.2%}")
        results = {"validation_accuracy": validation_metric.mean()}

        return results
