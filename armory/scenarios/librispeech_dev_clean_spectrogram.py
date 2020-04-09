"""
Classifier evaluation within ARMORY
"""

import logging
from importlib import import_module

import numpy as np

from armory.utils.config_loading import load_dataset, load_model
from armory import paths

from armory.utils import metrics
from armory.scenarios.base import Scenario

logger = logging.getLogger(__name__)


class LibrispeechDevCleanSpectrogram(Scenario):
    def _evaluate(self, config: dict) -> dict:
        """
        Evaluate a config file for classification robustness against attack.
        """
        model_config = config["model"]
        classifier, preprocessing_fn = load_model(model_config)

        n_tbins = 100  # number of time bins in spectrogram input to model

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

            val_data_generator = load_dataset(
                config["dataset"],
                epochs=fit_kwargs["nb_epochs"],
                split_type="validation",
                preprocessing_fn=preprocessing_fn,
            )

            for e in range(fit_kwargs["nb_epochs"]):
                logger.info("Epoch: {}/{}".format(e, fit_kwargs["nb_epochs"]))
                for _ in range(train_data_generator.batches_per_epoch):
                    x_train, y_train = train_data_generator.get_batch()

                    # x_train is of shape (N,241,T), representing N spectrograms,
                    # each with 241 frequency bins and T time bins that's variable,
                    # depending on the duration of the corresponding raw audio.
                    # The model accepts a fixed size spectrogram, so x_trains need to
                    # be sampled.

                    x_train_seg = []
                    for xt in x_train:
                        rand_t = np.random.randint(xt.shape[1] - n_tbins)
                        x_train_seg.append(xt[:, rand_t : rand_t + n_tbins])
                    x_train_seg = np.array(x_train_seg)
                    x_train_seg = np.expand_dims(x_train_seg, -1)
                    classifier.fit(
                        x_train_seg,
                        y_train,
                        batch_size=config["dataset"]["batch_size"],
                        nb_epochs=1,
                        **{"verbose": True},  # TODO: Arg
                    )

                # evaluate on validation examples
                correct = 0
                cnt = 0
                for _ in range(val_data_generator.batches_per_epoch):
                    x_val, y_val = val_data_generator.get_batch()
                    x_val_seg = []
                    y_val_seg = []
                    for xt, yt in zip(x_val, y_val):
                        n_seg = int(xt.shape[1] / n_tbins)
                        xt = xt[:, : n_seg * n_tbins]
                        for ii in range(n_seg):
                            x_val_seg.append(xt[:, ii * n_tbins : (ii + 1) * n_tbins])
                            y_val_seg.append(yt)
                    x_val_seg = np.array(x_val_seg)
                    x_val_seg = np.expand_dims(x_val_seg, -1)
                    y_val_seg = np.array(y_val_seg)

                    y = classifier.predict(x_val_seg)
                    correct += np.sum(np.argmax(y, 1) == y_val_seg)
                    cnt += len(y_val_seg)
                validation_acc = float(correct) / cnt
                logger.info("Validation accuracy: {}".format(validation_acc))

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

        task_metric = metrics.categorical_accuracy

        correct = 0
        cnt = 0
        benign_accuracies = []
        for _ in range(test_data_generator.batches_per_epoch):
            x_test, y_test = test_data_generator.get_batch()
            x_test_seg = []
            y_test_seg = []
            for xt, yt in zip(x_test, y_test):
                n_seg = int(xt.shape[1] / n_tbins)
                xt = xt[:, : n_seg * n_tbins]
                for ii in range(n_seg):
                    x_test_seg.append(xt[:, ii * n_tbins : (ii + 1) * n_tbins])
                    y_test_seg.append(yt)
            x_test_seg = np.array(x_test_seg)
            x_test_seg = np.expand_dims(x_test_seg, -1)
            y_test_seg = np.array(y_test_seg)

            y_pred = classifier.predict(x_test_seg)
            benign_accuracies.extend(task_metric(y_test_seg, y_pred))
        benign_accuracy = sum(benign_accuracies) / test_data_generator.size
        logger.info(f"Accuracy on benign test examples: {benign_accuracy:.2%}")

        # Evaluate the ART classifier on adversarial test examples
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
            x_test_seg = []
            y_test_seg = []
            for xt, yt in zip(x_test, y_test):
                n_seg = int(xt.shape[1] / n_tbins)
                xt = xt[:, : n_seg * n_tbins]
                for ii in range(n_seg):
                    x_test_seg.append(xt[:, ii * n_tbins : (ii + 1) * n_tbins])
                    y_test_seg.append(yt)
            x_test_seg = np.array(x_test_seg)
            x_test_seg = np.expand_dims(x_test_seg, -1)
            y_test_seg = np.array(y_test_seg)

            attack = attack_fn(
                classifier=classifier, **attack_config["kwargs"], batch_size=32
            )
            x_test_adv = attack.generate(x=x_test_seg)

            y = classifier.predict(x_test_adv)
            correct += np.sum(np.argmax(y, 1) == y_test_seg)
            cnt += len(y_test_seg)
        adv_acc = float(correct) / cnt
        logger.info("Adversarial accuracy: {}".format(adv_acc))

        logger.info("Saving json output...")
        logger.info(
            f"Evaluation Results written to {paths.docker().output_dir}/librispeech_spectrogram_evaluation-results.json"
        )
        results = {
            "mean_benign_accuracy": str(benign_accuracy),
            "mean_adversarial_accuracy": str(adv_acc),
        }
        return results
