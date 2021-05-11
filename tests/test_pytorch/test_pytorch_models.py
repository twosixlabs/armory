from importlib import import_module

import numpy as np
import pytest

from armory.data import datasets
from armory.utils.config_loading import load_dataset
from armory.data.utils import maybe_download_weights_from_s3
from armory import paths
from armory.utils.metrics import object_detection_AP_per_class

DATASET_DIR = paths.DockerPaths().dataset_dir


@pytest.mark.usefixtures("ensure_armory_dirs")
def test_pytorch_mnist():
    classifier_module = import_module("armory.baseline_models.pytorch.mnist")
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
def test_pytorch_mnist_pretrained():
    classifier_module = import_module("armory.baseline_models.pytorch.mnist")
    classifier_fn = getattr(classifier_module, "get_art_model")
    weights_path = maybe_download_weights_from_s3("undefended_mnist_5epochs.pth")
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
    classifier_module = import_module("armory.baseline_models.pytorch.cifar")
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
    assert (accuracy / test_dataset.batches_per_epoch) > 0.25


@pytest.mark.usefixtures("ensure_armory_dirs")
def test_pytorch_xview_pretrained():
    detector_module = import_module("armory.baseline_models.pytorch.xview_frcnn")
    detector_fn = getattr(detector_module, "get_art_model")
    weights_path = maybe_download_weights_from_s3(
        "xview_model_state_dict_epoch_99_loss_0p67"
    )
    detector = detector_fn(
        model_kwargs={}, wrapper_kwargs={}, weights_path=weights_path,
    )

    NUM_TEST_SAMPLES = 250
    dataset_config = {
        "batch_size": 1,
        "framework": "numpy",
        "module": "armory.data.datasets",
        "name": "xview",
    }
    test_dataset = load_dataset(
        dataset_config,
        epochs=1,
        split="test",
        num_batches=NUM_TEST_SAMPLES,
        shuffle_files=False,
    )

    list_of_ys = []
    list_of_ypreds = []
    for x, y in test_dataset:
        y_pred = detector.predict(x)
        list_of_ys.extend(y)
        list_of_ypreds.extend(y_pred)

    average_precision_by_class = object_detection_AP_per_class(
        list_of_ys, list_of_ypreds
    )
    mAP = np.fromiter(average_precision_by_class.values(), dtype=float).mean()
    for class_id in [4, 23, 33, 39]:
        assert average_precision_by_class[class_id] > 0.9
    assert mAP > 0.25


@pytest.mark.usefixtures("ensure_armory_dirs")
def test_pytorch_gtsrb():
    classifier_module = import_module("armory.baseline_models.pytorch.micronnet_gtsrb")
    classifier_fn = getattr(classifier_module, "get_art_model")
    preprocessing_fn = getattr(classifier_module, "preprocessing_fn")
    classifier = classifier_fn(model_kwargs={}, wrapper_kwargs={})

    train_dataset = datasets.german_traffic_sign(
        split="train",
        epochs=5,
        batch_size=128,
        dataset_dir=DATASET_DIR,
        preprocessing_fn=preprocessing_fn,
    )
    test_dataset = datasets.german_traffic_sign(
        split="test",
        epochs=1,
        batch_size=128,
        dataset_dir=DATASET_DIR,
        preprocessing_fn=preprocessing_fn,
    )

    classifier.fit_generator(
        train_dataset, nb_epochs=5,
    )

    accuracy = 0
    for _ in range(test_dataset.batches_per_epoch):
        x, y = test_dataset.get_batch()
        predictions = classifier.predict(x)
        accuracy += np.sum(np.argmax(predictions, axis=1) == y) / len(y)
    assert (accuracy / test_dataset.batches_per_epoch) > 0.8
