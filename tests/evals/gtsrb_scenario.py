"""
Classifier evaluation within ARMORY
"""

import json
import os
import sys
import logging
import coloredlogs
from importlib import import_module

import numpy as np
from tensorflow.keras.utils import to_categorical

from armory.utils.config_loading import load_dataset, load_model
from armory import paths

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
coloredlogs.install(logging.INFO)

def evaluate_classifier(config_path: str) -> None:
    """
    Evaluate a config file for classiifcation robustness against attack.
    """
    with open(config_path) as f:
        config = json.load(f)

    model_config = config["model"]
    classifier, preprocessing_fn = load_model(model_config)

    logger.info(f"Loading dataset {config['dataset']['name']}...")
    train_epochs = config["adhoc"]["train_epochs"]

    '''
    Train on training data - could be clean or poisoned
    and validation on clean data
    '''
    logger.info(
        f"Fitting model of {model_config['module']}.{model_config['name']}..."
    )
    for e in range(train_epochs):
        logger.info("Epoch: {}".format(e))

        train_data_generator = load_dataset(
            config["dataset"],
            epochs=train_epochs,
            split_type="train",
            preprocessing_fn=preprocessing_fn,
        )

        '''
        For poisoned dataset, change to validation_data_generator
        using split_type="val"
        '''
        test_data_generator = load_dataset(
            config["dataset"],
            epochs=1,
            split_type="test",
            preprocessing_fn=preprocessing_fn,
        )

        # train
        for _ in range(train_data_generator.batches_per_epoch):
            x_train, y_train = train_data_generator.get_batch()
            y_train = to_categorical(y_train, 43)
            classifier.fit(x_train, y_train, batch_size=len(y_train), nb_epochs=1, **{"verbose": False})

        # validate on clean data
        correct = 0
        cnt = 0
        for _ in range(test_data_generator.batches_per_epoch):
            x_test, y_test = test_data_generator.get_batch()
            y = classifier.predict(x_test)
            correct += np.sum(np.argmax(y, 1)==y_test)
            cnt += len(y_test)
        validation_accuracy = float(correct)/cnt
        logger.info("Validation accuracy: {}".format(validation_accuracy))

    # Evaluate on test examples - clean or poisoned
    test_data_generator = load_dataset(
        config["dataset"],
        epochs=1,
        split_type="test",
        preprocessing_fn=preprocessing_fn,
    )

    correct = 0
    cnt = 0
    for _ in range(test_data_generator.batches_per_epoch):
        x_test, y_test = test_data_generator.get_batch()
        y = classifier.predict(x_test)
        correct += np.sum(np.argmax(y, 1)==y_test)
        cnt += len(y_test)
    poison_accuracy = float(correct)/cnt
    logger.info("Test accuracy: {}".format(poison_accuracy))

    '''
    Generate poison examples
    Ignore this section if using existing poisoned dataset
    '''
    attack = import_module("art")
    print(attack.__version__)
    attack_config = config["attack"]
    attack_module = import_module(attack_config["module"])
    attack_fn = getattr(attack_module, attack_config["name"])
    poison_module = import_module(attack_attack_config["kwargs"]["poison_module"])
    poison_fn = getattr(poison_module, attack_config["kwargs"]["poison_type"])

    logger.info("Generating poisoning  examples...")
    test_data_generator = load_dataset(
        config["dataset"],
        epochs=1,
        split_type="test",
        preprocessing_fn=preprocessing_fn,
    )

    attack = attack_fn(poison_fn)

    '''
    In this example, all images of "src" class have a trigger
    added and re-labeled as "tgt" class
    '''
    src = 0
    tgt = 42
    poison_imgs = []
    poison_labels = []
    for _ in range(test_data_generator.batches_per_epoch):
        x_test, y_test = test_data_generator.get_batch()
        src_imgs = x_test[y_test == src]
        p_imgs, p_labels = attack.poison(src_imgs, tgt*np.ones(len(src_imgs)))
        poison_imgs.append(p_imgs)
        poison_labels.append(p_labels)
    poison_imgs = np.array(poison_imgs)
    poison_labels = np.array(poison_labels)
    print(poison_imgs.shape, poison_labels.shape)

    # Saving results
    logger.info("Saving json output...")
    filepath = os.path.join(paths.docker().output_dir, "gtsrb-evaluation-results.json")
    with open(filepath, "w") as f:
        output_dict = {
            "config": config,
            "results": {
                "validation_accuracy": str(validation_accuracy),
                "poison_accuracy": str(poison_accuracy),
            },
        }
        json.dump(output_dict, f, sort_keys=True, indent=4)
    logger.info(f"Evaluation Results written <output_dir>/evaluation-results.json")

if __name__ == "__main__":
    config_path = sys.argv[-1]
    evaluate_classifier(config_path)
