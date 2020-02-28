"""
Test cases for ARMORY datasets.
"""

import unittest

import numpy as np
from importlib import import_module

from armory.data import datasets
from armory import paths


class DatasetTest(unittest.TestCase):
    def test_mnist(self):
        train_x, train_y, test_x, test_y = datasets.mnist(
            dataset_dir=paths.host().dataset_dir
        )
        self.assertEqual(train_x.shape[0], 60000)
        self.assertEqual(train_y.shape[0], 60000)
        self.assertEqual(test_x.shape[0], 10000)
        self.assertEqual(test_y.shape[0], 10000)

    def test_cifar10(self):
        train_x, train_y, test_x, test_y = datasets.cifar10(
            dataset_dir=paths.host().dataset_dir
        )
        self.assertEqual(train_x.shape[0], 50000)
        self.assertEqual(train_y.shape[0], 50000)
        self.assertEqual(test_x.shape[0], 10000)
        self.assertEqual(test_y.shape[0], 10000)

    def test_digit(self):
        train_x, train_y, test_x, test_y = datasets.digit(
            dataset_dir=paths.host().dataset_dir
        )
        self.assertEqual(train_x.shape[0], 1350)
        self.assertEqual(train_y.shape[0], 1350)
        self.assertEqual(test_x.shape[0], 150)
        self.assertEqual(test_y.shape[0], 150)
        for X in train_x, test_x:
            for x in X:
                self.assertTrue(1148 <= len(x) <= 18262)

    def test_imagenet_adv(self):
        clean_x, adv_x, labels = datasets.imagenet_adversarial(
            dataset_dir=paths.host().dataset_dir
        )
        self.assertEqual(clean_x.shape[0], 1000)
        self.assertEqual(adv_x.shape[0], 1000)
        self.assertEqual(labels.shape[0], 1000)

    def test_german_traffic_sign(self):
        train_x, train_y, test_x, test_y = datasets.german_traffic_sign(
            dataset_dir=paths.host().dataset_dir
        )
        self.assertEqual(train_x.shape[0], 39209)
        self.assertEqual(train_y.shape[0], 39209)
        self.assertEqual(test_x.shape[0], 12630)
        self.assertEqual(test_y.shape[0], 12630)
        for X in train_x, test_x:
            for x in X:
                self.assertTrue(25 <= x.shape[0] <= 232)
                self.assertTrue(25 <= x.shape[1] <= 266)


class KerasTest(unittest.TestCase):
    def test_keras_mnist(self):
        batch_size = 64
        epochs = 2

        classifier_module = import_module("armory.baseline_models.keras.keras_mnist")
        classifier_fn = getattr(classifier_module, "get_art_model")
        classifier = classifier_fn(model_kwargs={}, wrapper_kwargs={})
        preprocessing_fn = getattr(classifier_module, "preprocessing_fn")

        train_x, train_y, test_x, test_y = datasets.mnist(
            preprocessing_fn=preprocessing_fn, dataset_dir=paths.host().dataset_dir,
        )

        classifier.fit(train_x, train_y, batch_size=batch_size, nb_epochs=epochs)

        predictions = classifier.predict(test_x)
        accuracy = np.sum(np.argmax(predictions, axis=1) == test_y) / len(test_y)
        self.assertGreater(accuracy, 0.95)

    def test_keras_cifar10(self):
        batch_size = 64
        epochs = 2

        classifier_module = import_module("armory.baseline_models.keras.keras_cifar")
        classifier_fn = getattr(classifier_module, "get_art_model")
        classifier = classifier_fn(model_kwargs={}, wrapper_kwargs={})
        preprocessing_fn = getattr(classifier_module, "preprocessing_fn")

        train_x, train_y, test_x, test_y = datasets.cifar10(
            preprocessing_fn=preprocessing_fn, dataset_dir=paths.host().dataset_dir,
        )

        classifier.fit(train_x, train_y, batch_size=batch_size, nb_epochs=epochs)

        predictions = classifier.predict(test_x)
        accuracy = np.sum(np.argmax(predictions, axis=1) == test_y) / len(test_y)
        self.assertGreater(accuracy, 0.4)

    def test_keras_imagenet(self):
        classifier_module = import_module("armory.baseline_models.keras.keras_resnet50")
        classifier_fn = getattr(classifier_module, "get_art_model")
        classifier = classifier_fn(model_kwargs={}, wrapper_kwargs={})
        preprocessing_fn = getattr(classifier_module, "preprocessing_fn")

        clean_x, adv_x, labels = datasets.imagenet_adversarial(
            preprocessing_fn=preprocessing_fn, dataset_dir=paths.host().dataset_dir,
        )

        predictions = classifier.predict(clean_x)
        accuracy = np.sum(np.argmax(predictions, axis=1) == labels) / len(labels)
        self.assertGreater(accuracy, 0.65)

        predictions = classifier.predict(adv_x)
        accuracy = np.sum(np.argmax(predictions, axis=1) == labels) / len(labels)
        self.assertLess(accuracy, 0.02)

    def test_keras_imagenet_transfer(self):
        classifier_module = import_module(
            "armory.baseline_models.keras.keras_inception_resnet_v2"
        )
        classifier_fn = getattr(classifier_module, "get_art_model")
        classifier = classifier_fn(model_kwargs={}, wrapper_kwargs={})
        preprocessing_fn = getattr(classifier_module, "preprocessing_fn")

        clean_x, adv_x, labels = datasets.imagenet_adversarial(
            preprocessing_fn=preprocessing_fn, dataset_dir=paths.host().dataset_dir,
        )

        predictions = classifier.predict(clean_x)
        accuracy = np.sum(np.argmax(predictions, axis=1) == labels) / len(labels)
        self.assertGreater(accuracy, 0.75)

        predictions = classifier.predict(adv_x)
        accuracy = np.sum(np.argmax(predictions, axis=1) == labels) / len(labels)
        self.assertLess(accuracy, 0.72)
