"""

"""

import unittest
from armory.webapi.data import SUPPORTED_DATASETS
from importlib import import_module


class DatasetTest(unittest.TestCase):
    def test_batching(self):
        batch_size = 48
        train_ds, test_ds, num_train, num_test = SUPPORTED_DATASETS["mnist"](
            batch_size=batch_size, epochs=1
        )
        for x in train_ds:
            self.assertEqual(x[0].shape[0], batch_size)
            self.assertEqual(x[1].shape[0], batch_size)

        self.assertEqual(test_ds[0].shape[0], 10000)
        self.assertEqual(test_ds[1].shape[0], 10000)

    def test_mnist(self):
        batch_size = 64
        epochs = 2
        train_ds, test_ds, num_train, num_test = SUPPORTED_DATASETS["mnist"](
            batch_size=batch_size, epochs=epochs
        )
        self.assertEqual(num_train, 60000)
        self.assertEqual(num_test, 10000)

    def test_cifar10(self):
        batch_size = 64
        epochs = 2
        train_ds, test_ds, num_train, num_test = SUPPORTED_DATASETS["cifar10"](
            batch_size=batch_size, epochs=epochs
        )
        self.assertEqual(num_train, 50000)
        self.assertEqual(num_test, 10000)


class KerasTest(unittest.TestCase):
    def test_keras_mnist(self):
        batch_size = 64
        epochs = 2
        train_ds, test_ds, num_train, _ = SUPPORTED_DATASETS["mnist"](
            batch_size=batch_size, epochs=epochs
        )

        classifier_module = import_module("armory.baseline_models.keras.keras_mnist")
        classifier = getattr(classifier_module, "MODEL")

        steps_per_epoch = int(num_train / batch_size)
        history = classifier._model.fit_generator(
            train_ds, epochs=epochs, steps_per_epoch=steps_per_epoch,
        )

        self.assertGreater(history.history["acc"][-1], 0.95)

    def test_keras_cifar10(self):
        batch_size = 64
        epochs = 2
        train_ds, test_ds, num_train, _ = SUPPORTED_DATASETS["cifar10"](
            batch_size=batch_size, epochs=epochs
        )

        classifier_module = import_module("armory.baseline_models.keras.keras_cifar")
        classifier = getattr(classifier_module, "MODEL")

        steps_per_epoch = int(num_train / batch_size)
        history = classifier._model.fit_generator(
            train_ds, epochs=epochs, steps_per_epoch=steps_per_epoch,
        )

        self.assertGreater(history.history["acc"][-1], 0.4)
