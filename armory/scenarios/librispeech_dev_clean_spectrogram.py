"""
Classifier evaluation within ARMORY
"""

import logging

import numpy as np
from tqdm import tqdm

from armory.utils.config_loading import (
    load_dataset,
    load_model,
    load_attack,
)
from armory.utils import metrics
from armory.scenarios.base import Scenario

logger = logging.getLogger(__name__)


def segment(x, y, n_time_bins):
    """
    Return segmented batch of spectrograms and labels

    x is of shape (N,241,T), representing N spectrograms, each with 241 frequency bins
    and T time bins that's variable, depending on the duration of the corresponding
    raw audio.

    The model accepts a fixed size spectrogram, so data needs to be segmented for a
    fixed number of time_bins.
    """

    x_seg, y_seg = [], []
    for xt, yt in zip(x, y):
        n_seg = int(xt.shape[1] / n_time_bins)
        xt = xt[:, : n_seg * n_time_bins]
        for ii in range(n_seg):
            x_seg.append(xt[:, ii * n_time_bins : (ii + 1) * n_time_bins])
            y_seg.append(yt)
    x_seg = np.array(x_seg)
    x_seg = np.expand_dims(x_seg, -1)
    y_seg = np.array(y_seg)
    return x_seg, y_seg


class LibrispeechDevCleanSpectrogram(Scenario):
    def _evaluate(self, config: dict) -> dict:
        """
        Evaluate a config file for classification robustness against attack.
        """
        model_config = config["model"]
        classifier, preprocessing_fn = load_model(model_config)

        n_tbins = 100  # number of time bins in spectrogram input to model

        task_metric = metrics.categorical_accuracy

        # Train ART classifier
        if not model_config["weights_file"]:
            classifier.set_learning_phase(True)
            logger.info(
                f"Fitting model {model_config['module']}.{model_config['name']}..."
            )
            fit_kwargs = model_config["fit_kwargs"]
            train_data_generator = load_dataset(
                config["dataset"],
                epochs=fit_kwargs["nb_epochs"],
                split_type="train",
                preprocessing_fn=preprocessing_fn,
            )

            for cnt, (x, y) in tqdm(enumerate(train_data_generator)):
                x_seg, y_seg = segment(x, y, n_tbins)
                classifier.fit(
                    x_seg,
                    y_seg,
                    batch_size=config["dataset"]["batch_size"],
                    nb_epochs=1,
                    verbose=True,
                )

                if (cnt + 1) % train_data_generator.batches_per_epoch == 0:
                    # evaluate on validation examples
                    val_data_generator = load_dataset(
                        config["dataset"],
                        epochs=1,
                        split_type="validation",
                        preprocessing_fn=preprocessing_fn,
                    )

                    cnt = 0
                    validation_accuracies = []
                    for x_val, y_val in tqdm(val_data_generator):
                        x_val_seg, y_val_seg = segment(x_val, y_val, n_tbins)
                        y_pred = classifier.predict(x_val_seg)
                        validation_accuracies.extend(task_metric(y_val_seg, y_pred))
                        cnt += len(y_val_seg)
                    validation_accuracy = sum(validation_accuracies) / cnt
                    logger.info("Validation accuracy: {}".format(validation_accuracy))

        classifier.set_learning_phase(False)
        # Evaluate ART classifier on test examples
        logger.info(f"Loading testing dataset {config['dataset']['name']}...")
        test_data_generator = load_dataset(
            config["dataset"],
            epochs=1,
            split_type="test",
            preprocessing_fn=preprocessing_fn,
        )

        logger.info("Running inference on benign test examples...")

        cnt = 0
        benign_accuracies = []
        for x, y in tqdm(test_data_generator, desc="Benign"):
            x_seg, y_seg = segment(x, y, n_tbins)
            y_pred = classifier.predict(x_seg)
            benign_accuracies.extend(task_metric(y_seg, y_pred))
            cnt += len(y_seg)

        benign_accuracy = sum(benign_accuracies) / cnt
        logger.info(f"Accuracy on benign test examples: {benign_accuracy:.2%}")

        # Evaluate the ART classifier on adversarial test examples
        logger.info("Generating / testing adversarial examples...")
        attack = load_attack(config["attack"], classifier)

        test_data_generator = load_dataset(
            config["dataset"],
            epochs=1,
            split_type="test",
            preprocessing_fn=preprocessing_fn,
        )

        cnt = 0
        adversarial_accuracies = []
        for x, y in tqdm(test_data_generator, desc="Attack"):
            x_seg, y_seg = segment(x, y, n_tbins)
            x_adv = attack.generate(x=x_seg)
            y_pred = classifier.predict(x_adv)
            adversarial_accuracies.extend(task_metric(y_seg, y_pred))
            cnt += len(y_seg)
        adversarial_accuracy = sum(adversarial_accuracies) / cnt
        logger.info(
            f"Accuracy on adversarial test examples: {adversarial_accuracy:.2%}"
        )

        results = {
            "mean_benign_accuracy": benign_accuracy,
            "mean_adversarial_accuracy": adversarial_accuracy,
        }
        return results
