from importlib import import_module

import numpy as np
import pytest

from armory.data import datasets
from armory.data import adversarial_datasets
from armory.data.utils import maybe_download_weights_from_s3
from armory import paths

DATASET_DIR = paths.runtime_paths().dataset_dir


@pytest.mark.usefixtures("ensure_armory_dirs")
def test_keras_mnist():
    classifier_module = import_module("armory.baseline_models.keras.mnist")
    classifier_fn = getattr(classifier_module, "get_art_model")
    classifier = classifier_fn(model_kwargs={}, wrapper_kwargs={})

    train_dataset = datasets.mnist(
        split="train", epochs=1, batch_size=600, dataset_dir=DATASET_DIR,
    )
    test_dataset = datasets.mnist(
        split="test", epochs=1, batch_size=100, dataset_dir=DATASET_DIR,
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
def test_keras_mnist_pretrained():
    classifier_module = import_module("armory.baseline_models.keras.mnist")
    classifier_fn = getattr(classifier_module, "get_art_model")
    weights_path = maybe_download_weights_from_s3("undefended_mnist_5epochs.h5")
    classifier = classifier_fn(
        model_kwargs={}, wrapper_kwargs={}, weights_path=weights_path
    )

    test_dataset = datasets.mnist(
        split="test", epochs=1, batch_size=100, dataset_dir=DATASET_DIR,
    )

    accuracy = 0
    for _ in range(test_dataset.batches_per_epoch):
        x, y = test_dataset.get_batch()
        predictions = classifier.predict(x)
        accuracy += np.sum(np.argmax(predictions, axis=1) == y) / len(y)
    assert (accuracy / test_dataset.batches_per_epoch) > 0.98


@pytest.mark.usefixtures("ensure_armory_dirs")
def test_keras_cifar():
    classifier_module = import_module("armory.baseline_models.keras.cifar")
    classifier_fn = getattr(classifier_module, "get_art_model")
    classifier = classifier_fn(model_kwargs={}, wrapper_kwargs={})

    train_dataset = datasets.cifar10(
        split="train", epochs=1, batch_size=500, dataset_dir=DATASET_DIR,
    )
    test_dataset = datasets.cifar10(
        split="test", epochs=1, batch_size=100, dataset_dir=DATASET_DIR,
    )

    classifier.fit_generator(
        train_dataset, nb_epochs=1,
    )

    accuracy = 0
    for _ in range(test_dataset.batches_per_epoch):
        x, y = test_dataset.get_batch()
        predictions = classifier.predict(x)
        accuracy += np.sum(np.argmax(predictions, axis=1) == y) / len(y)
    assert (accuracy / test_dataset.batches_per_epoch) > 0.27


@pytest.mark.usefixtures("ensure_armory_dirs")
def test_keras_imagenet():
    classifier_module = import_module("armory.baseline_models.keras.resnet50")
    classifier_fn = getattr(classifier_module, "get_art_model")
    weights_path = maybe_download_weights_from_s3("resnet50_imagenet_v1.h5")
    classifier = classifier_fn(
        model_kwargs={}, wrapper_kwargs={}, weights_path=weights_path
    )

    dataset = adversarial_datasets.imagenet_adversarial(
        split="adversarial", epochs=1, batch_size=100, dataset_dir=DATASET_DIR,
    )

    accuracy_clean = 0
    accuracy_adv = 0
    for _ in range(dataset.batches_per_epoch):
        (x_clean, x_adv), y = dataset.get_batch()
        predictions_clean = classifier.predict(x_clean)
        accuracy_clean += np.sum(np.argmax(predictions_clean, axis=1) == y) / len(y)
        predictions_adv = classifier.predict(x_adv)
        accuracy_adv += np.sum(np.argmax(predictions_adv, axis=1) == y) / len(y)
    assert (accuracy_clean / dataset.batches_per_epoch) > 0.65
    assert (accuracy_adv / dataset.batches_per_epoch) < 0.02


@pytest.mark.usefixtures("ensure_armory_dirs")
def test_keras_imagenet_transfer():
    classifier_module = import_module(
        "armory.baseline_models.keras.inception_resnet_v2"
    )
    classifier_fn = getattr(classifier_module, "get_art_model")
    weights_path = maybe_download_weights_from_s3("inceptionresnetv2_imagenet_v1.h5")
    classifier = classifier_fn(
        model_kwargs={}, wrapper_kwargs={}, weights_path=weights_path
    )

    dataset = adversarial_datasets.imagenet_adversarial(
        split="adversarial", epochs=1, batch_size=100, dataset_dir=DATASET_DIR,
    )
    accuracy_clean = 0
    accuracy_adv = 0
    for _ in range(dataset.batches_per_epoch):
        (x_clean, x_adv), y = dataset.get_batch()
        predictions_clean = classifier.predict(x_clean)
        accuracy_clean += np.sum(np.argmax(predictions_clean, axis=1) == y) / len(y)
        predictions_adv = classifier.predict(x_adv)
        accuracy_adv += np.sum(np.argmax(predictions_adv, axis=1) == y) / len(y)

    assert (accuracy_clean / dataset.batches_per_epoch) > 0.74
    assert (accuracy_adv / dataset.batches_per_epoch) < 0.73
