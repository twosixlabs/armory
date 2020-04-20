from importlib import import_module

import numpy as np
import pytest

from armory.data import datasets
from armory import paths

DATASET_DIR = paths.docker().dataset_dir


@pytest.mark.usefixtures("ensure_armory_dirs")
def test_pytorch_mnist():
    classifier_module = import_module("armory.baseline_models.pytorch.mnist")
    classifier_fn = getattr(classifier_module, "get_art_model")
    classifier = classifier_fn(model_kwargs={}, wrapper_kwargs={})
    preprocessing_fn = getattr(classifier_module, "preprocessing_fn")

    train_dataset = datasets.mnist(
        split_type="train",
        epochs=1,
        batch_size=600,
        dataset_dir=DATASET_DIR,
        preprocessing_fn=preprocessing_fn,
    )
    test_dataset = datasets.mnist(
        split_type="test",
        epochs=1,
        batch_size=100,
        dataset_dir=DATASET_DIR,
        preprocessing_fn=preprocessing_fn,
    )

    classifier.fit_generator(
        train_dataset, nb_epochs=1,
    )

    accuracy = 0
    for _ in range(test_dataset.batches_per_epoch):
        x, y = test_dataset.get_batch()
        predictions = classifier.predict(x)
        accuracy += np.sum(np.argmax(predictions, axis=1) == y) / len(y)
    assert (accuracy / test_dataset.batches_per_epoch) > 0.9


@pytest.mark.usefixtures("ensure_armory_dirs")
def test_pytorch_mnist_pretrained():
    classifier_module = import_module("armory.baseline_models.pytorch.mnist")
    classifier_fn = getattr(classifier_module, "get_art_model")
    classifier = classifier_fn(
        model_kwargs={}, wrapper_kwargs={}, weights_file="undefended_mnist_5epochs.pth"
    )
    preprocessing_fn = getattr(classifier_module, "preprocessing_fn")

    test_dataset = datasets.mnist(
        split_type="test",
        epochs=1,
        batch_size=100,
        dataset_dir=DATASET_DIR,
        preprocessing_fn=preprocessing_fn,
    )

    accuracy = 0
    for _ in range(test_dataset.batches_per_epoch):
        x, y = test_dataset.get_batch()
        predictions = classifier.predict(x)
        accuracy += np.sum(np.argmax(predictions, axis=1) == y) / len(y)
    assert (accuracy / test_dataset.batches_per_epoch) > 0.98


@pytest.mark.usefixtures("ensure_armory_dirs")
def test_keras_cifar():
    classifier_module = import_module("armory.baseline_models.pytorch.cifar")
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
        train_dataset, nb_epochs=1,
    )

    accuracy = 0
    for _ in range(test_dataset.batches_per_epoch):
        x, y = test_dataset.get_batch()
        predictions = classifier.predict(x)
        accuracy += np.sum(np.argmax(predictions, axis=1) == y) / len(y)
    assert (accuracy / test_dataset.batches_per_epoch) > 0.25
