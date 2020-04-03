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
import time

from tensorflow.keras.utils import to_categorical
import tensorflow.keras
from armory.utils.config_loading import load_dataset, load_model
from armory import paths

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
coloredlogs.install(logging.DEBUG)

def evaluate_classifier(config_path: str) -> None:
    """
    Evaluate a config file for classiifcation robustness against attack.
    """
    logger.info(tensorflow.keras.__version__)
    with open(config_path) as f:
        config = json.load(f)
    model_config = config["model"]
    model_status = model_config['model_kwargs']['model_status'] # "trained"/"untrained"
    logger.info(model_status)
    classifier, preprocessing_fn = load_model(model_config)

    n_tbins = 100 # numbef of time bins in spectrogram input to model
    """
    Mapping origin Librispeech labels, which represent 9000+ speakers,
    into 40 speakers that comprise the dev_clean dataset
    """
    class_map = {84: 0, 174: 1, 251: 2, 422: 3, 652: 4, 777: 5,
             1272: 6, 1462: 7, 1673: 8, 1919: 9, 1988: 10,
             1993: 11, 2035: 12, 2078: 13, 2086: 14, 2277: 15,
             2412: 16, 2428: 17, 2803: 18, 2902: 19, 3000: 20,
             3081: 21, 3170: 22, 3536: 23, 3576: 24, 3752: 25,
             3853: 26, 5338: 27, 5536: 28, 5694: 29, 5895: 30,
             6241: 31, 6295: 32, 6313: 33, 6319: 34, 6345: 35,
             7850: 36, 7976: 37, 8297: 38, 8842: 39}

    # Train ART classifier
    if model_status == 'untrained':
        logger.info(f"Fitting clean model of {model_config['module']}.{model_config['name']}...")
        logger.info(f"Loading training dataset {config['dataset']['name']}...")
        train_epochs = config["adhoc"]["train_epochs"]
        batch_size = config['dataset']['batch_size']
        train_data_generator = load_dataset(
            config["dataset"],
            epochs=train_epochs,
            split_type="train",
            preprocessing_fn=preprocessing_fn,
        )

        for e in range(train_epochs):
            logger.info("Epoch: {}/{}".format(e, train_epochs))
            for _ in range(train_data_generator.batches_per_epoch):
                x_train, y_train = train_data_generator.get_batch()
                y_train = np.array([class_map[y] for y in y_train])
                """
                x_train is of shape (N,241,T), representing N spectrograms,
                each with 241 frequency bins and T time bins that's variable,
                depending on the duration of the corresponding raw audio.
                The model accepts a fixed size spectrogram, so x_trains need to
                be sampled.
                """
                x_train_seg = []
                for xt in x_train:
                    rand_t = np.random.randint(xt.shape[1]-n_tbins)
                    x_train_seg.append(xt[:,rand_t:rand_t+n_tbins])
                x_train_seg = np.array(x_train_seg)
                x_train_seg = np.expand_dims(x_train_seg, -1)
                y_train_onehot = to_categorical(y_train, num_classes=40)
                classifier.fit(x_train_seg, y_train_onehot, batch_size=batch_size, nb_epochs=1,
                               **{"verbose":True})

            # evaluate on validation examples
            val_data_generator = load_dataset(
                config["dataset"],
                epochs=1,
                split_type="validation",
                preprocessing_fn=preprocessing_fn,
            )

            correct = 0
            cnt = 0
            for _ in range(val_data_generator.batches_per_epoch):
                x_val, y_val = val_data_generator.get_batch()
                y_val = np.array([class_map[y] for y in y_val])
                x_val_seg = []
                y_val_seg = []
                for xt, yt in zip(x_val, y_val):
                    n_seg = int(xt.shape[1]/n_tbins)
                    xt = xt[:,:n_seg*n_tbins]
                    for ii in range(n_seg):
                       x_val_seg.append(xt[:,ii*n_tbins:(ii+1)*n_tbins])
                       y_val_seg.append(yt)
                x_val_seg = np.array(x_val_seg)
                x_val_seg = np.expand_dims(x_val_seg, -1)
                y_val_seg = np.array(y_val_seg)

                y = classifier.predict(x_val_seg)
                correct += np.sum(np.argmax(y,1) == y_val_seg)
                cnt += len(y_val_seg)
            validation_acc = float(correct)/cnt
            logger.info("Validation accuracy: {}".format(validation_acc))

    # Evaluate ART classifier on test examples
    logger.info("Running inference on benign test examples...")
    logger.info(f"Loading testing dataset {config['dataset']['name']}...")
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
        y_test = np.array([class_map[y] for y in y_test])
        x_test_seg = []
        y_test_seg = []
        for xt, yt in zip(x_test, y_test):
            n_seg = int(xt.shape[1]/n_tbins)
            xt = xt[:,:n_seg*n_tbins]
            for ii in range(n_seg):
                x_test_seg.append(xt[:,ii*n_tbins:(ii+1)*n_tbins])
                y_test_seg.append(yt)
        x_test_seg = np.array(x_test_seg)
        x_test_seg = np.expand_dims(x_test_seg, -1)
        y_test_seg = np.array(y_test_seg)

        y = classifier.predict(x_test_seg)
        correct += np.sum(np.argmax(y,1) == y_test_seg)
        cnt += len(y_test_seg)
    test_acc = float(correct)/cnt
    logger.info("Test accuracy: {}".format(test_acc))

    ## Evaluate the ART classifier on adversarial test examples
    logger.info("Generating / testing adversarial examples...")
    attack_config = config["attack"]
    attack_module = import_module(attack_config["module"])
    attack_fn = getattr(attack_module, attack_config["name"])

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
        y_test = np.array([class_map[y] for y in y_test])
        x_test_seg = []
        y_test_seg = []
        for xt, yt in zip(x_test, y_test):
            n_seg = int(xt.shape[1]/n_tbins)
            xt = xt[:,:n_seg*n_tbins]
            for ii in range(n_seg):
                x_test_seg.append(xt[:,ii*n_tbins:(ii+1)*n_tbins])
                y_test_seg.append(yt)
        x_test_seg = np.array(x_test_seg)
        x_test_seg = np.expand_dims(x_test_seg, -1)
        y_test_seg = np.array(y_test_seg)

        logger.info(x_test_seg.shape[0])

        attack = attack_fn(classifier=classifier, **attack_config["kwargs"], batch_size=32)
        x_test_adv = attack.generate(x=x_test_seg)

        y = classifier.predict(x_test_adv)
        correct += np.sum(np.argmax(y,1) == y_test_seg)
        cnt += len(y_test_seg)
    adv_acc = float(correct)/cnt
    logger.info("Adversarial accuracy: {}".format(adv_acc))

    logger.info("Saving json output...")
    filepath = os.path.join(paths.docker().output_dir, "librispeech_spectrogram_evaluation-results.json")
    with open(filepath, "w") as f:
        output_dict = {
            "config": config,
            "results": {
                "baseline_accuracy": str(test_acc),
                "adversarial_accuracy": str(adv_acc),
            },
        }
        json.dump(output_dict, f, sort_keys=True, indent=4)
    logger.info(f"Evaluation Results written to {paths.docker().output_dir}/librispeech_spectrogram_evaluation-results.json")

if __name__ == "__main__":
    config_path = sys.argv[-1]
    evaluate_classifier(config_path)
