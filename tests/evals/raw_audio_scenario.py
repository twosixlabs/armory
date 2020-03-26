"""
Classifier evaluation within ARMORY
"""

import json
import os
import sys
import logging
from importlib import import_module

import numpy as np

from armory.utils.config_loading import load_dataset, load_model
from armory import paths
from collections import Counter

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


DEMO = True

LABEL_MAP = [84, 174, 251, 422, 652, 777, 1272, 1462, 1673, 1919, 1988, 1993, 2035, 2078, 2086, 2277, 2412, 2428, 2803, 2902, 3000, 3081, 3170, 3536, 3576, 3752, 3853, 5338, 5536, 5694, 5895, 6241, 6295, 6313, 6319, 6345, 7850, 7976, 8297, 8842]

def evaluate_classifier(config_path: str) -> None:
    with open(config_path) as f:
        config = json.load(f)

    model_config = config["model"]    
    classifier, preprocessing_fn = load_model(model_config)

    logger.info(f"Loading dataset {config['dataset']['name']}...")
    
    # set to 1 for now, should be 2900
    test_data_generator = load_dataset(
        config["dataset"],
        epochs=1,
        split_type="test",
        preprocessing_fn=preprocessing_fn
    )    
    
    if(not model_config["model_kwargs"]["pretrained"]):
        logger.info(
            f"Fitting clean unpoisoned model of {model_config['module']}.{model_config['name']}..."
        )
        train_data_generator = load_dataset(
            config["dataset"],
            epochs=config["adhoc"]["epochs"],
            split_type="train",
            preprocessing_fn=preprocessing_fn,
        )

        if DEMO:
            nb_epochs = 10
        else:
            nb_epochs = train_data_generator.total_iterations
        # TODO [immediate] index out of bounds for fit. using label as an index for an array that has nb_classes elements 
        #(determined by get_art_model, in PyTorchClassifier declaration in sincnet.py)
        # generator needs to output label in range (0-39)
        classifier.fit_generator(train_data_generator, nb_epochs=nb_epochs)

    # Evaluate the ART classifier on benign test examples
    logger.info("Running inference on benign examples...")
    benign_accuracy = 0
    cnt = 0

    if DEMO:
        iterations = 640
    else:
        iterations = test_data_generator.total_iterations // 2
    for _ in range(iterations):
        x, y = test_data_generator.get_batch()
        predictions = classifier.predict(x) 
        benign_accuracy += np.sum([LABEL_MAP[i] for i in np.argmax(predictions, axis=1)] == y) / len(y)
        cnt += 1
    logger.info(
        "Accuracy on benign test examples: {}%".format(benign_accuracy * 100 / cnt)
    )
    
    adversarial_accuracy = 0 # TEMP until adversarial dataset is created

    logger.info("Saving json output...")
    filepath = os.path.join(paths.docker().output_dir, "evaluation-results.json")
    with open(filepath, "w") as f:
        output_dict = {
            "config": config,
            "results": {
                "baseline_accuracy": str(benign_accuracy),
                "adversarial_accuracy": str(adversarial_accuracy),
            },
        }
        json.dump(output_dict, f, sort_keys=True, indent=4)
    logger.info(f"Evaluation Results written <output_dir>/evaluation-results.json")
    

if __name__ == "__main__":
    config_path = sys.argv[-1]
    evaluate_classifier(config_path)
