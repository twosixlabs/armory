"""
Classifier evaluation within ARMORY
"""

from importlib import import_module
import json
import logging
import sys

from art.attacks import CarliniL2Method, CarliniLInfMethod

from armory.webapi.data import SUPPORTED_DATASETS

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def evaluate_classifier(config_path: str) -> None:
    """
    Evaluate a config file for classiifcation robustness against attack.
    """
    batch_size = 64
    epochs = 3

    with open(config_path, "r") as fp:
        config = json.load(fp)

    train_ds, test_ds, num_train, num_test = SUPPORTED_DATASETS[config["data"]](
        batch_size=batch_size, epochs=epochs, normalize=True
    )
    x_test, y_test = test_ds

    classifier_module = import_module(config["model_file"])
    classifier = getattr(classifier_module, config["model_name"])

    steps_per_epoch = int(num_train / batch_size)
    classifier._model.fit_generator(
        train_ds, epochs=epochs, steps_per_epoch=steps_per_epoch
    )

    # Evaluate the ART classifier on benign test examples
    benign_accuracy = classifier._model.evaluate(x_test, y_test, batch_size=batch_size)
    logger.info(
        "Accuracy on benign test examples: {}%".format(benign_accuracy[1] * 100)
    )

    # Generate adversarial test examples
    if config["attack"] == "CarliniL2Method":
        attack = CarliniL2Method(
            classifier=classifier,
            targeted=True,
            learning_rate=0.01,
            binary_search_steps=10,
            max_iter=10,
            initial_const=0.01,
            max_halving=5,
            max_doubling=5,
            batch_size=1,
        )
        attack_shrt = "L2"
    elif config["attack"] == "CarliniLInfMethod":
        attack = CarliniLInfMethod(
            classifier,
            confidence=0.0,
            targeted=True,
            learning_rate=0.01,
            max_iter=10,
            max_halving=5,
            max_doubling=5,
            eps=0.3,
            batch_size=128,
        )
        attack_shrt = "LInf"
    else:
        raise ValueError(
            f"Invalid attack {config['attack']}: only CarliniL2Method and CarliniInfMethod supported"
        )

    num_classes = 10
    num_attacked_pts = 100
    y_target = (y_test + 1) % num_classes

    x_test_adv = attack.generate(
        x=x_test[:num_attacked_pts], y=y_target[:num_attacked_pts]
    )

    # Evaluate the ART classifier on adversarial test examples
    targeted_attack_success_rate = classifier._model.evaluate(
        x_test_adv[:num_attacked_pts], y_target[:num_attacked_pts], batch_size=64
    )

    clean_accuracy = classifier._model.evaluate(
        x_test[num_attacked_pts:], y_test[num_attacked_pts:]
    )

    results = dict()
    results["targeted_attack_success_rate"] = targeted_attack_success_rate[1] * 100
    results["clean_input_classification_accuracy"] = clean_accuracy[1] * 100

    filepath = f"outputs/carlini_wagner_attack_{attack_shrt}_targeted_output.json"
    with open(filepath, "w") as f:
        output_dict = {
            "config": config,
            "results": results,
        }
        json.dump(output_dict, f, sort_keys=True, indent=4)


if __name__ == "__main__":
    config_path = sys.argv[-1]
    evaluate_classifier(config_path)
