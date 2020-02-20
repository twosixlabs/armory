"""
Classifier evaluation within ARMORY
"""

from importlib import import_module
import json
import logging
import os
import shutil
import sys
import time

import numpy as np

from armory.eval.plot_poisoning import classification_poisoning
from armory.utils.config_loading import load_dataset
from armory.paths import DockerPaths

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def init_metrics(pct_poisons, n_trials):
    return {pct_poison: [None for _ in range(n_trials)] for pct_poison in pct_poisons}


def eval_targeted_fit(classifier, x, y):
    y_pred = np.argmax(classifier.predict(x), axis=1)
    return np.sum(y_pred == y) / float(y_pred.shape[0])


def summarize_metrics(raw_metrics):
    """
    Computes means and standard deviations of trial-level metrics
    :param raw_metrics: dictionary indexed by metric, trial number and percent poisoning (if relevant)
    :return: dictionary indexed by summary metric and percent poisoning (if relevant)
    """
    summarized_metrics = {}
    for metric_name, value in raw_metrics.items():
        if isinstance(value, dict):
            # Metric that varies with the percentage of poisons and by trials
            summarized_metrics[metric_name + "_mean"] = {}
            summarized_metrics[metric_name + "_std"] = {}
            for pct_poison, value2 in value.items():
                summarized_metrics[metric_name + "_mean"][pct_poison] = np.mean(value2)
                summarized_metrics[metric_name + "_std"][pct_poison] = np.std(value2)

        if isinstance(value, list):
            # Metric only varies by trials
            summarized_metrics[metric_name + "_mean"] = np.mean(value)
            summarized_metrics[metric_name + "_std"] = np.std(value)
    return summarized_metrics


def evaluate_classifier(config_path: str) -> None:
    """
    Evaluate a config file for classifcation robustness against attack.
    """
    # Generate adversarial test examples
    docker_paths = DockerPaths()

    with open(config_path, "r") as fp:
        config = json.load(fp)

    model_config = config["model"]
    classifier_module = import_module(model_config["module"])
    classifier_fn = getattr(classifier_module, model_config["name"])
    preprocessing_fn = getattr(classifier_module, "preprocessing_fn")

    logger.info(f"Loading dataset {config['dataset']['name']}...")
    x_clean_train, y_clean_train, x_clean_test, y_clean_test = load_dataset(
        config["dataset"], preprocessing_fn=preprocessing_fn
    )

    batch_size = config["adhoc"]["batch_size"]
    epochs = config["adhoc"]["epochs"]
    n_trials = config["adhoc"]["n_trials"]
    poison_frac_min = config["adhoc"]["poison_frac_min"]
    poison_frac_max = config["adhoc"]["poison_frac_max"]
    poison_frac_steps = config["adhoc"]["poison_frac_steps"]
    source_class = config["adhoc"]["source_class"]
    target_class = config["adhoc"]["target_class"]

    fraction_poisons = np.linspace(poison_frac_min, poison_frac_max, poison_frac_steps)

    # Test clean model accuracy to provide a benchmark to poisoned model accuracy
    raw_metrics = {}
    raw_metrics["undefended_backdoor_success_rate"] = init_metrics(
        fraction_poisons, n_trials
    )
    raw_metrics["non_backdoored_accuracy"] = init_metrics(fraction_poisons, n_trials)
    raw_metrics["clean_model_accuracy"] = [None for _ in range(n_trials)]
    raw_metrics["defended_backdoor_success_rate"] = init_metrics(
        fraction_poisons, n_trials
    )
    raw_metrics["delta_accuracy"] = init_metrics(fraction_poisons, n_trials)

    for trial in range(n_trials):
        classifier = classifier_fn(
            model_config["model_kwargs"], model_config["wrapper_kwargs"]
        )
        logger.info(
            f"Fitting clean unpoisoned model of {model_config['module']}.{model_config['name']}..."
        )
        classifier.fit(
            x_clean_train, y_clean_train, batch_size=batch_size, nb_epochs=epochs
        )
        raw_metrics["clean_model_accuracy"][trial] = eval_targeted_fit(
            classifier, x_clean_test, y_clean_test
        )

        for frac_poison in fraction_poisons:
            # Need to retrain from scratch for each frac_poison value
            classifier = classifier_fn(
                model_config["model_kwargs"], model_config["wrapper_kwargs"]
            )
            classifier_defended = classifier_fn(
                model_config["model_kwargs"], model_config["wrapper_kwargs"]
            )

            attack_config = config["attack"]
            attack_module = import_module(attack_config["module"])
            attack_fn = getattr(attack_module, attack_config["name"])

            attack = attack_fn(
                classifier=classifier,
                x_train=x_clean_train,
                y_train=y_clean_train,
                pct_poison=frac_poison,
                source_class=source_class,
                target_class=target_class,
            )

            is_poison, x_poison, y_poison = attack.generate(
                x_clean_train, y_clean_train
            )
            logger.info(f"Fitting poisoned model with poison fraction {frac_poison}...")
            classifier.fit(x_poison, y_poison, batch_size=batch_size, nb_epochs=epochs)

            x_test_targeted = x_clean_test[y_clean_test == source_class]
            x_poison_test = attack.generate_target_test(x_test_targeted)

            # Show targeted accuracy for poisoned classes is as expected
            raw_metrics["undefended_backdoor_success_rate"][frac_poison][
                trial
            ] = eval_targeted_fit(classifier, x_poison_test, target_class)

            raw_metrics["non_backdoored_accuracy"][frac_poison][
                trial
            ] = eval_targeted_fit(classifier, x_clean_test, y_clean_test)

            defense_config = config["defense"]
            defense_module = import_module(defense_config["module"])
            defense_fn = getattr(defense_module, defense_config["name"])

            defense = defense_fn(
                classifier,
                x_poison,
                y_poison,
                batch_size=batch_size,
                ub_pct_poison=frac_poison,
                **defense_config["kwargs"],
            )
            conf_matrix_json = defense.evaluate_defence(np.logical_not(is_poison))
            logger.info(
                f"Poison detection confusion matrix from defense {config['defense']['name']} "
                f"with poison fraction {frac_poison}:"
            )
            logger.info(conf_matrix_json)
            _, indices_to_keep = defense.detect_poison()

            logger.info(
                f"Fitting poisoned model with poisons filtered by defense {config['defense']['name']} "
                f"with poison fraction {frac_poison}..."
            )
            classifier_defended.fit(
                x_poison[indices_to_keep == 1],
                y_poison[indices_to_keep == 1],
                batch_size=batch_size,
                nb_epochs=epochs,
            )

            defended_backdoor_success_rate = eval_targeted_fit(
                classifier_defended, x_poison_test, target_class
            )
            raw_metrics["defended_backdoor_success_rate"][frac_poison][
                trial
            ] = defended_backdoor_success_rate
            logger.info(
                f"Trial {trial+1} defended backdoor success rate {defended_backdoor_success_rate} "
                f"with poisoning proportion of {frac_poison}"
            )

            defended_clean_accuracy = eval_targeted_fit(
                classifier_defended, x_clean_test, y_clean_test
            )

            delta_accuracy = (
                raw_metrics["non_backdoored_accuracy"][frac_poison][trial]
                - defended_clean_accuracy
            )
            raw_metrics["delta_accuracy"][frac_poison][trial] = delta_accuracy

            logger.info(
                f"Trial {trial+1} delta accuracy of {delta_accuracy} "
                f"with poisoning proportion of {frac_poison}"
            )
        logger.info(f"Trial {trial+1}/{n_trials} completed.")

    summarized_metrics = summarize_metrics(raw_metrics)
    logger.info("Saving json output...")
    filepath = os.path.join(
        docker_paths.output_dir, f"backdoor_performance_{int(time.time())}.json"
    )
    with open(filepath, "w") as f:
        output_dict = {"config": config, "results": summarized_metrics}
        json.dump(output_dict, f, sort_keys=True, indent=4)
    shutil.copyfile(filepath, os.path.join(docker_paths.output_dir, "latest.json"))
    classification_poisoning(filepath)
    classification_poisoning(os.path.join(docker_paths.output_dir, "latest.json"))


if __name__ == "__main__":
    config_path = sys.argv[-1]
    evaluate_classifier(config_path)
