"""
Test cases for ARMORY datasets.
"""

import unittest

import numpy as np
from importlib import import_module

from armory.data import datasets
from armory import paths


class DatasetTest(unittest.TestCase):
    def test_mnist_train(self):
        train_dataset = datasets.mnist(
            split_type="train",
            epochs=1,
            batch_size=600,
            dataset_dir=paths.host().dataset_dir,
        )
        self.assertEqual(train_dataset.size, 60000)
        self.assertEqual(train_dataset.batch_size, 600)
        self.assertEqual(train_dataset.total_iterations, 100)

        x, y = train_dataset.get_batch()
        self.assertEqual(x.shape, (600, 28, 28, 1))
        self.assertEqual(y.shape, (600,))

    def test_mnist_test(self):
        test_dataset = datasets.mnist(
            split_type="test",
            epochs=1,
            batch_size=100,
            dataset_dir=paths.host().dataset_dir,
        )
        self.assertEqual(test_dataset.size, 10000)
        self.assertEqual(test_dataset.batch_size, 100)
        self.assertEqual(test_dataset.total_iterations, 100)

        x, y = test_dataset.get_batch()
        self.assertEqual(x.shape, (100, 28, 28, 1))
        self.assertEqual(y.shape, (100,))

    def test_cifar_train(self):
        train_dataset = datasets.cifar10(
            split_type="train",
            epochs=1,
            batch_size=500,
            dataset_dir=paths.host().dataset_dir,
        )
        self.assertEqual(train_dataset.size, 50000)
        self.assertEqual(train_dataset.batch_size, 500)
        self.assertEqual(train_dataset.total_iterations, 100)

        x, y = train_dataset.get_batch()
        self.assertEqual(x.shape, (500, 32, 32, 3))
        self.assertEqual(y.shape, (500,))

    def test_cifar_test(self):
        test_dataset = datasets.cifar10(
            split_type="test",
            epochs=1,
            batch_size=100,
            dataset_dir=paths.host().dataset_dir,
        )
        self.assertEqual(test_dataset.size, 10000)
        self.assertEqual(test_dataset.batch_size, 100)
        self.assertEqual(test_dataset.total_iterations, 100)

        x, y = test_dataset.get_batch()
        self.assertEqual(x.shape, (100, 32, 32, 3))
        self.assertEqual(y.shape, (100,))

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
        classifier_module = import_module("armory.baseline_models.keras.keras_mnist")
        classifier_fn = getattr(classifier_module, "get_art_model")
        classifier = classifier_fn(model_kwargs={}, wrapper_kwargs={})
        preprocessing_fn = getattr(classifier_module, "preprocessing_fn")

        train_dataset = datasets.mnist(
            split_type="train",
            epochs=1,
            batch_size=600,
            dataset_dir=paths.host().dataset_dir,
            preprocessing_fn=preprocessing_fn,
        )
        test_dataset = datasets.mnist(
            split_type="test",
            epochs=1,
            batch_size=100,
            dataset_dir=paths.host().dataset_dir,
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
        self.assertGreater(accuracy / test_dataset.total_iterations, 0.9)

    def test_keras_cifar(self):
        classifier_module = import_module("armory.baseline_models.keras.keras_cifar")
        classifier_fn = getattr(classifier_module, "get_art_model")
        classifier = classifier_fn(model_kwargs={}, wrapper_kwargs={})
        preprocessing_fn = getattr(classifier_module, "preprocessing_fn")

        train_dataset = datasets.cifar10(
            split_type="train",
            epochs=1,
            batch_size=500,
            dataset_dir=paths.host().dataset_dir,
            preprocessing_fn=preprocessing_fn,
        )
        test_dataset = datasets.cifar10(
            split_type="test",
            epochs=1,
            batch_size=100,
            dataset_dir=paths.host().dataset_dir,
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
