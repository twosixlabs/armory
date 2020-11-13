from importlib import import_module

import numpy as np
import pytest

from armory.data import datasets, adversarial_datasets
from armory import paths
from armory.utils.metrics import (
    object_detection_AP_per_class,
    apricot_patch_targeted_AP_per_class,
)

DATASET_DIR = paths.runtime_paths().dataset_dir


@pytest.mark.usefixtures("ensure_armory_dirs")
def test_tf1_mnist():
    classifier_module = import_module("armory.baseline_models.tf_graph.mnist")
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
def test_tf1_apricot():
    detector_module = import_module("armory.baseline_models.tf_graph.mscoco_frcnn")
    detector_fn = getattr(detector_module, "get_art_model")
    detector = detector_fn(model_kwargs={}, wrapper_kwargs={})

    test_dataset = adversarial_datasets.apricot_dev_adversarial(
        split="adversarial",
        epochs=1,
        batch_size=1,
        dataset_dir=DATASET_DIR,
        shuffle_files=False,
    )

    list_of_ys = []
    list_of_ypreds = []
    for x, y in test_dataset:
        y_pred = detector.predict(x)
        list_of_ys.append(y)
        list_of_ypreds.append(y_pred)

    average_precision_by_class = object_detection_AP_per_class(
        list_of_ys, list_of_ypreds
    )
    mAP = np.fromiter(average_precision_by_class.values(), dtype=float).mean()
    for class_id in [13, 15, 64]:
        assert average_precision_by_class[class_id] > 0.79
    assert mAP > 0.08

    patch_targeted_AP_by_class = apricot_patch_targeted_AP_per_class(
        list_of_ys, list_of_ypreds
    )
    expected_patch_targeted_AP_by_class = {
        1: 0.18,
        17: 0.18,
        27: 0.27,
        33: 0.51,
        44: 0.14,
    }
    for class_id, expected_AP in expected_patch_targeted_AP_by_class.items():
        assert np.abs(patch_targeted_AP_by_class[class_id] - expected_AP) < 0.03
