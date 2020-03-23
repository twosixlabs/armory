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

def get_label_map():
    metadata_dir = os.path.join(paths.docker().dataset_dir, "librispeech_dev_clean_split/plain_text/1.1.0/metadata.json")
    metadata = json.load(open(metadata_dir))    
    relevant_speakers = {}
    for id, info in metadata["speakers"].items():
        if(info["subset"] == "dev-clean"):
            relevant_speakers[id] = info
    original_id_list = list(map(int,relevant_speakers.keys()))
    original_id_list.sort()
    return original_id_list


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
        # TODO train here

    # Evaluate the ART classifier on benign test examples
    logger.info("Running inference on benign examples...")
    benign_accuracy = 0
    cnt = 0
    label_map = get_label_map()
    if DEMO:
        iterations = 640
    else:
        iterations = test_data_generator.total_iterations // 2
    for _ in range(iterations):
        x, y = test_data_generator.get_batch()
        predictions = classifier.predict(x) 
        benign_accuracy += np.sum([label_map[i] for i in np.argmax(predictions, axis=1)] == y) / len(y)
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
