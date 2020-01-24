"""
Classifier evaluation within ARMORY
"""

from importlib import import_module
import json
import logging
import shutil
import sys
import time

import numpy as np

from armory.eval.plot_poisoning import classification_poisoning
from armory.data.data import SUPPORTED_DATASETS
from armory.art_experimental import attacks as attacks_extended

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def evaluate_classifier(config_path: str) -> None:
    """
    Evaluate a config file for classiifcation robustness against attack.
    """
    # Generate adversarial test examples

    batch_size = 64
    epochs = 10

    with open(config_path, "r") as fp:
        config = json.load(fp)

    classifier_module = import_module(config["model_file"])
    preprocessing_fn = getattr(classifier_module, "preprocessing_fn")

    x_clean_train, y_clean_train, x_clean_test, y_clean_test = SUPPORTED_DATASETS[
        config["data"]
    ](preprocessing_fn=preprocessing_fn)

    if config["adversarial_budget"]["percent_poisons"] != "all":
        raise NotImplementedError("Script runs only for all epsilon.")

    source_class = config["source_class"]
    target_class = config["target_class"]

    pct_poisons = np.linspace(0.1, 0.9, 9)
    backdoor_success_rates = []
    delta_accuracies = []

    # Test clean model accuracy to provide a benchmark to poisoned model accuracy
    classifier = classifier_module.make_new_model()
    classifier.fit(
        x_clean_train, y_clean_train, batch_size=batch_size, nb_epochs=epochs
    )
    y_pred = classifier.predict(x_clean_test)
    clean_model_accuracy = np.sum(np.argmax(y_pred, axis=1) == y_clean_test) / len(
        y_clean_test
    )
    logger.info(f"Clean model accuracy {clean_model_accuracy}")

    for pct_poison in pct_poisons:
        # Need to retrain from scratch for each pct_poison value
        classifier = classifier_module.make_new_model()

        attack = attacks_extended.PoisoningAttackBackdoor(
            classifier,
            x_clean_train,
            y_clean_train,
            pct_poison,
            source_class,
            target_class,
        )

        is_poison, x_poison, y_poison = attack.generate(x_clean_train, y_clean_train)

        classifier.fit(x_poison, y_poison, batch_size=batch_size, nb_epochs=epochs)

        x_test_targeted = x_clean_test[y_clean_test == source_class]

        x_poison_test = attack.generate_target_test(x_test_targeted)

        # Show targeted accuracy for poisoned classes is as expected
        y_pred_flat = np.argmax(classifier.predict(x_poison_test), axis=1)

        backdoor_success_rate = np.sum(y_pred_flat == target_class) / float(
            y_pred_flat.shape[0]
        )
        backdoor_success_rates.append(backdoor_success_rate)

        y_pred = classifier.predict(x_clean_test)
        non_backdoored_accuracy = np.sum(
            np.argmax(y_pred, axis=1) == y_clean_test
        ) / len(y_clean_test)
        delta_accuracies.append(non_backdoored_accuracy - clean_model_accuracy)

        logger.info(f"Backdoor success rate {backdoor_success_rate}")
        logger.info(f"Non-backdoored accuracy {non_backdoored_accuracy}")

    results = {
        "backdoor_success_rate": backdoor_success_rates,
        "clean_model_accuracy": clean_model_accuracy,
        "delta_accuracy": delta_accuracies,
        "percent_poisons": pct_poisons.tolist(),
    }

    filepath = f"outputs/backdoor_performance_{int(time.time())}.json"
    with open(filepath, "w") as f:
        output_dict = {"config": config, "results": results}
        json.dump(output_dict, f, sort_keys=True, indent=4)
    shutil.copyfile(filepath, "outputs/latest.json")
    classification_poisoning(filepath)
    classification_poisoning("outputs/latest.json")


if __name__ == "__main__":
    config_path = sys.argv[-1]
    evaluate_classifier(config_path)
