"""
Test cases for ARMORY datasets.
"""

import os
import unittest

from armory.data import datasets
from armory import paths

DATASET_DIR = paths.docker().dataset_dir


class DatasetTest(unittest.TestCase):
    def test_mnist_train(self):
        train_dataset = datasets.mnist(
            split_type="train", epochs=1, batch_size=600, dataset_dir=DATASET_DIR,
        )
        self.assertEqual(train_dataset.size, 60000)
        self.assertEqual(train_dataset.batch_size, 600)
        self.assertEqual(train_dataset.total_iterations, 100)

        x, y = train_dataset.get_batch()
        self.assertEqual(x.shape, (600, 28, 28, 1))
        self.assertEqual(y.shape, (600,))

    def test_mnist_test(self):
        test_dataset = datasets.mnist(
            split_type="test", epochs=1, batch_size=100, dataset_dir=DATASET_DIR,
        )
        self.assertEqual(test_dataset.size, 10000)
        self.assertEqual(test_dataset.batch_size, 100)
        self.assertEqual(test_dataset.total_iterations, 100)

        x, y = test_dataset.get_batch()
        self.assertEqual(x.shape, (100, 28, 28, 1))
        self.assertEqual(y.shape, (100,))

    def test_cifar_train(self):
        train_dataset = datasets.cifar10(
            split_type="train", epochs=1, batch_size=500, dataset_dir=DATASET_DIR,
        )
        self.assertEqual(train_dataset.size, 50000)
        self.assertEqual(train_dataset.batch_size, 500)
        self.assertEqual(train_dataset.total_iterations, 100)

        x, y = train_dataset.get_batch()
        self.assertEqual(x.shape, (500, 32, 32, 3))
        self.assertEqual(y.shape, (500,))

    def test_cifar_test(self):
        test_dataset = datasets.cifar10(
            split_type="test", epochs=1, batch_size=100, dataset_dir=DATASET_DIR,
        )
        self.assertEqual(test_dataset.size, 10000)
        self.assertEqual(test_dataset.batch_size, 100)
        self.assertEqual(test_dataset.total_iterations, 100)

        x, y = test_dataset.get_batch()
        self.assertEqual(x.shape, (100, 32, 32, 3))
        self.assertEqual(y.shape, (100,))

    def test_digit(self):
        epochs = 1
        batch_size = 1
        num_users = 3
        min_length = 1148
        max_length = 18262
        for split, size in [
            ("train", 45 * num_users * 10),
            ("test", 5 * num_users * 10),
        ]:
            test_dataset = datasets.digit(
                split_type=split,
                epochs=epochs,
                batch_size=batch_size,
                dataset_dir=DATASET_DIR,
            )
            self.assertEqual(test_dataset.size, size)
            self.assertEqual(test_dataset.batch_size, batch_size)

            x, y = test_dataset.get_batch()
            self.assertEqual(x.shape[0], batch_size)
            self.assertEqual(x.ndim, 2)
            self.assertTrue(min_length <= x.shape[1] <= max_length)
            self.assertEqual(y.shape, (batch_size,))

    def test_imagenet_adv(self):
        test_dataset = datasets.imagenet_adversarial(
            dataset_dir=DATASET_DIR, split_type="clean", batch_size=100, epochs=1,
        )
        self.assertEqual(test_dataset.size, 1000)
        self.assertEqual(test_dataset.batch_size, 100)
        self.assertEqual(test_dataset.total_iterations, 10)

        x, y = test_dataset.get_batch()
        self.assertEqual(x.shape, (100, 224, 224, 3))
        self.assertEqual(y.shape, (100,))

    def test_german_traffic_sign(self):
        for split, size in [("train", 39209), ("test", 12630)]:
            batch_size = 1
            epochs = 1
            test_dataset = datasets.german_traffic_sign(
                split_type=split,
                epochs=epochs,
                batch_size=batch_size,
                dataset_dir=DATASET_DIR,
            )
            self.assertEqual(test_dataset.size, size)

            x, y = test_dataset.get_batch()
            # sign image shape is variable so we don't compare 2nd dim
            self.assertEqual(x.shape[:1] + x.shape[3:], (batch_size, 3))
            self.assertEqual(y.shape, (batch_size,))

    def test_imagenette(self):
        if not os.path.isdir(
            os.path.join(DATASET_DIR, "imagenette", "full-size", "0.1.0")
        ):
            self.skipTest("imagenette dataset not locally available.")

        for split, size in [("train", 12894), ("validation", 500)]:
            batch_size = 1
            epochs = 1
            test_dataset = datasets.imagenette(
                split_type=split,
                epochs=epochs,
                batch_size=batch_size,
                dataset_dir=DATASET_DIR,
            )
            self.assertEqual(test_dataset.size, size)

            x, y = test_dataset.get_batch()
            # image dimensions are variable so we don't compare 2nd dim or 3rd dim
            self.assertEqual(x.shape[:1] + x.shape[3:], (batch_size, 3))
            self.assertEqual(y.shape, (batch_size,))

    def test_ucf101(self):
        if not os.path.isdir(os.path.join(DATASET_DIR, "ucf101", "ucf101_1", "2.0.0")):
            self.skipTest("ucf101 dataset not locally available.")

        for split, size in [("train", 9537), ("test", 3783)]:
            batch_size = 1
            epochs = 1
            test_dataset = datasets.ucf101(
                split_type=split,
                epochs=epochs,
                batch_size=batch_size,
                dataset_dir=DATASET_DIR,
            )
            self.assertEqual(test_dataset.size, size)

            x, y = test_dataset.get_batch()
            # video length is variable so we don't compare 2nd dim
            self.assertEqual(x.shape[:1] + x.shape[2:], (batch_size, 240, 320, 3))
            self.assertEqual(y.shape, (batch_size,))

    def test_librispeech(self):
        if not os.path.exists(os.path.join(DATASET_DIR, "librispeech_dev_clean_split")):
            self.skipTest("Librispeech dataset not downloaded.")

        splits = ("train", "validation", "test")
        sizes = (1371, 692, 640)
        min_dim1s = (23120, 26239, 24080)
        max_dim1s = (519760, 516960, 522320)
        batch_size = 1

        for split, size, min_dim1, max_dim1 in zip(splits, sizes, min_dim1s, max_dim1s):
            train_dataset = datasets.librispeech_dev_clean(
                split_type=split,
                epochs=1,
                batch_size=batch_size,
                dataset_dir=DATASET_DIR,
            )
            self.assertEqual(train_dataset.size, size)
            self.assertEqual(train_dataset.batch_size, batch_size)
            self.assertEqual(
                train_dataset.total_iterations,
                size // batch_size + bool(size % batch_size),
            )

            x, y = train_dataset.get_batch()
            self.assertEqual(x.shape[0], 1)
            self.assertTrue(min_dim1 <= x.shape[1] <= max_dim1)
            self.assertEqual(y.shape, (batch_size,))

    def test_resisc45(self):
        """
        Skip test if not locally available
        """
        if not os.path.isdir(os.path.join(DATASET_DIR, "resisc45_split", "3.0.0")):
            self.skipTest("resisc45_split dataset not locally available.")

        for split, size in [("train", 22500), ("validation", 4500), ("test", 4500)]:
            batch_size = 16
            epochs = 1
            test_dataset = datasets.resisc45(
                split_type=split,
                epochs=epochs,
                batch_size=batch_size,
                dataset_dir=DATASET_DIR,
            )
            self.assertEqual(test_dataset.size, size)
            self.assertEqual(test_dataset.batch_size, batch_size)
            self.assertEqual(
                test_dataset.total_iterations,
                size // batch_size + bool(size % batch_size),
            )

            x, y = test_dataset.get_batch()
            self.assertEqual(x.shape, (batch_size, 256, 256, 3))
            self.assertEqual(y.shape, (batch_size,))

    def test_variable_length(self):
        """
        Test batches with variable length items using digit dataset
        """
        size = 1350
        batch_size = 4
        test_dataset = datasets.digit(
            split_type="train",
            epochs=1,
            batch_size=batch_size,
            dataset_dir=DATASET_DIR,
        )
        self.assertEqual(
            test_dataset.total_iterations, size // batch_size + bool(size % batch_size)
        )

        x, y = test_dataset.get_batch()
        self.assertEqual(x.dtype, object)
        self.assertEqual(x.shape, (batch_size,))
        for x_i in x:
            self.assertEqual(x_i.ndim, 1)
            self.assertTrue(1148 <= len(x_i) <= 18262)
        self.assertEqual(y.shape, (batch_size,))
