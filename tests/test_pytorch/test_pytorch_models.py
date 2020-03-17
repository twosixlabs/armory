import unittest
from importlib import import_module

import numpy as np

from armory.data import datasets
from armory import paths

DATASET_DIR = paths.docker().dataset_dir


class PyTorchModelsTest(unittest.TestCase):
    def test_pytorch_mnist(self):
        pass

    def test_keras_cifar(self):
        classifier_module = import_module(
            "armory.baseline_models.pytorch.pytorch_cifar"
        )
        classifier_fn = getattr(classifier_module, "get_art_model")
        classifier = classifier_fn(model_kwargs={}, wrapper_kwargs={})
        preprocessing_fn = getattr(classifier_module, "preprocessing_fn")

        train_dataset = datasets.cifar10(
            split_type="train",
            epochs=1,
            batch_size=500,
            dataset_dir=DATASET_DIR,
            preprocessing_fn=preprocessing_fn,
        )
        test_dataset = datasets.cifar10(
            split_type="test",
            epochs=1,
            batch_size=100,
            dataset_dir=DATASET_DIR,
            preprocessing_fn=preprocessing_fn,
        )

        classifier.fit_generator(
            train_dataset, nb_epochs=train_dataset.total_iterations,
        )

        accuracy = 0
        for _ in range(test_dataset.total_iterations):
            x, y = test_dataset.get_batch()
            predictions = classifier.predict(x)
            accuracy += np.sum(np.argmax(predictions, axis=1) == y) / len(y)
        self.assertGreater(accuracy / test_dataset.total_iterations, 0.3)
