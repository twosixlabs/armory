"""
Classifier evaluation within ARMORY

Scenario Contributor: MITRE Corporation
"""

import logging

import numpy as np
from tqdm import tqdm

from armory.utils.config_loading import (
    load_dataset,
    load_model,
    load_attack,
    load_defense_wrapper,
    load_defense_internal,
    load,
)
from armory.utils import metrics
from armory.scenarios.base import Scenario

logger = logging.getLogger(__name__)


class Ucf101(Scenario):
    def _evaluate(self, config: dict) -> dict:
        """
        Evaluate the config and return a results dict
        """

        model_config = config["model"]
        classifier, preprocessing_fn = load_model(model_config)

        defense_config = config.get("defense") or {}
        defense_type = defense_config.get("type")

        IMAGE_DEFENSE_SUPPORT_ONLY_4D = True

        if defense_type in ["Preprocessor", "Postprocessor"]:
            if IMAGE_DEFENSE_SUPPORT_ONLY_4D:
                # Architectures that use 3D convolution takes 5D
                # inputs, usually in the shape of (nb, c, t, h, w)
                # To use image-based defense that support 4D inputs,
                # need to manually reshape 5D inputs into 4D, process
                # defense, and then reshape back to 5D
                defense = load(config.get("defense"))
            else:
                logger.info(f"Applying internal {defense_type} defense to classifier")
                classifier = load_defense_internal(config["defense"], classifier)

        if model_config["fit"]:
            classifier.set_learning_phase(True)
            logger.info(
                f"Fitting model {model_config['module']}.{model_config['name']}..."
            )
            train_epochs = config["model"]["fit_kwargs"]["nb_epochs"]
            batch_size = config["dataset"]["batch_size"]

            logger.info(f"Loading train dataset {config['dataset']['name']}...")
            train_data = load_dataset(
                config["dataset"],
                epochs=train_epochs,
                split_type="train",
                preprocessing_fn=preprocessing_fn,
            )

            if defense_type == "Trainer":
                logger.info(f"Training with {defense_type} defense...")
                defense = load_defense_wrapper(config["defense"], classifier)
            else:
                logger.info(f"Fitting classifier on clean train dataset...")

            for epoch in range(train_epochs):
                classifier.set_learning_phase(True)

                for _ in tqdm(
                    range(train_data.batches_per_epoch),
                    desc=f"Epoch: {epoch}/{train_epochs}",
                ):
                    x, y = train_data.get_batch()
                    # x_trains consists of one or more videos, each represented as an
                    # ndarray of shape (n_stacks, 3, 16, 112, 112).
                    # To train, randomly sample a batch of stacks
                    x = np.stack([x_i[np.random.randint(x_i.shape[0])] for x_i in x])
                    if defense_type == "Trainer":
                        defense.fit(x, y, batch_size=batch_size, nb_epochs=1)
                    else:
                        # apply defense outside of classifier
                        if IMAGE_DEFENSE_SUPPORT_ONLY_4D:
                            for ns in range(x.shape[0]):
                                xt = np.transpose(x[ns, ...], (1, 2, 3, 0))
                                if defense_config.get("name") == "JpegCompression":
                                    # jpegcompression requires unnormalized inputs
                                    xt[..., 0] += 114.7748
                                    xt[..., 1] += 107.7354
                                    xt[..., 2] += 99.4750
                                xt, _ = defense(xt)
                                if defense_config.get("name") == "JpegCompression":
                                    xt[..., 0] -= 114.7748
                                    xt[..., 1] -= 107.7354
                                    xt[..., 2] -= 99.4750
                                x[ns, ...] = np.transpose(xt, (3, 0, 1, 2))

                        classifier.fit(x, y, batch_size=batch_size, nb_epochs=1)

        if defense_type == "Transform":
            # NOTE: Transform currently not supported
            logger.info(f"Transforming classifier with {defense_type} defense...")
            defense = load_defense_wrapper(config["defense"], classifier)
            classifier = defense()

        classifier.set_learning_phase(False)

        # Evaluate the ART classifier on benign test examples
        logger.info(f"Loading test dataset {config['dataset']['name']}...")
        test_data_generator = load_dataset(
            config["dataset"],
            epochs=1,
            split_type="test",
            preprocessing_fn=preprocessing_fn,
        )

        logger.info("Running inference on benign examples...")
        metrics_logger = metrics.MetricsLogger.from_config(config["metric"])

        for x_batch, y_batch in tqdm(test_data_generator, desc="Benign"):
            for x, y in zip(x_batch, y_batch):

                # apply defense
                if IMAGE_DEFENSE_SUPPORT_ONLY_4D and defense_type in [
                    "Preprocessor",
                    "Postprocessor",
                ]:
                    for ns in range(x.shape[0]):
                        xt = np.transpose(x[ns, ...], (1, 2, 3, 0))
                        if defense_config.get("name") == "JpegCompression":
                            # jpegcompression requires unnormalized inputs
                            xt[..., 0] += 114.7748
                            xt[..., 1] += 107.7354
                            xt[..., 2] += 99.4750
                        xt, _ = defense(xt)
                        if defense_config.get("name") == "JpegCompression":
                            xt[..., 0] -= 114.7748
                            xt[..., 1] -= 107.7354
                            xt[..., 2] -= 99.4750
                        x[ns, ...] = np.transpose(xt, (3, 0, 1, 2))

                # combine predictions across all stacks
                y_pred = np.mean(classifier.predict(x), axis=0)
                metrics_logger.update_task(y, y_pred)
        metrics_logger.log_task()

        # Evaluate the ART classifier on adversarial test examples
        logger.info("Loading / testing adversarial examples...")

        if config["attack"]["preloaded"]:  # load existing adversarial dataset
            test_data_generator = load_dataset(
                config["attack"]["preloaded"],
                epochs=1,
                split_type="adversarial",
                preprocessing_fn=preprocessing_fn,
            )
        else:  # generate attack using ART
            attack = load_attack(config["attack"], classifier)
            test_data_generator = load_dataset(
                config["dataset"],
                epochs=1,
                split_type="test",
                preprocessing_fn=preprocessing_fn,
            )
        for x_batch, y_batch in tqdm(test_data_generator, desc="Attack"):
            if config["attack"]["preloaded"]:
                attack_type = config["attack"]["preloaded"]["attack_type"]
                x_clean_batch = x_batch["clean"]
                x_batch = x_batch[attack_type]
            for i, (x, y) in enumerate(zip(x_batch, y_batch)):
                if config["attack"]["preloaded"]:
                    x_adv = x
                    x_clean = x_clean_batch[i]
                else:
                    # each x is of shape (n_stack, 3, 16, 112, 112)
                    # n_stack varies
                    attack.set_params(batch_size=x.shape[0])
                    x_adv = attack.generate(x=x)
                    x_clean = x

                # calculate distance metrics before applying defense
                metrics_logger.update_perturbation([x_clean], [x_adv])

                # apply defense outside of classifier
                if IMAGE_DEFENSE_SUPPORT_ONLY_4D and defense_type in [
                    "Preprocessor",
                    "Postprocessor",
                ]:
                    for ns in range(x_adv.shape[0]):
                        xt = np.transpose(x_adv[ns, ...], (1, 2, 3, 0))
                        if defense_config.get("name") == "JpegCompression":
                            # jpegcompression requires unnormalized inputs
                            xt[..., 0] += 114.7748
                            xt[..., 1] += 107.7354
                            xt[..., 2] += 99.4750
                        xt, _ = defense(xt)
                        if defense_config.get("name") == "JpegCompression":
                            xt[..., 0] -= 114.7748
                            xt[..., 1] -= 107.7354
                            xt[..., 2] -= 99.4750
                        x_adv[ns, ...] = np.transpose(xt, (3, 0, 1, 2))

                # combine predictions across all stacks
                y_pred = np.mean(classifier.predict(x_adv), axis=0)
                metrics_logger.update_task(y, y_pred, adversarial=True)

        metrics_logger.log_task(adversarial=True)
        return metrics_logger.results()
