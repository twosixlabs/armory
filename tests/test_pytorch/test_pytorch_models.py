from importlib import import_module

import numpy as np
import pytest

from armory.data import datasets
from armory.data.utils import maybe_download_weights_from_s3
from armory import paths
from armory.utils.metrics import _object_detection_get_tp_fp_fn

DATASET_DIR = paths.DockerPaths().dataset_dir


@pytest.mark.usefixtures("ensure_armory_dirs")
def test_pytorch_mnist():
    classifier_module = import_module("armory.baseline_models.pytorch.mnist")
    classifier_fn = getattr(classifier_module, "get_art_model")
    classifier = classifier_fn(model_kwargs={}, wrapper_kwargs={})

    train_dataset = datasets.mnist(
        split_type="train", epochs=1, batch_size=600, dataset_dir=DATASET_DIR,
    )
    test_dataset = datasets.mnist(
        split_type="test", epochs=1, batch_size=100, dataset_dir=DATASET_DIR,
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
        split_type="test", epochs=1, batch_size=100, dataset_dir=DATASET_DIR,
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
        split_type="train", epochs=1, batch_size=500, dataset_dir=DATASET_DIR,
    )
    test_dataset = datasets.cifar10(
        split_type="test", epochs=1, batch_size=100, dataset_dir=DATASET_DIR,
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

    test_dataset = datasets.xview(
        split_type="test",
        epochs=1,
        batch_size=1,
        dataset_dir=DATASET_DIR,
        shuffle_files=False,
    )

    num_test_samples = 100
    tp_count = 0
    fp_count = 0
    fn_count = 0
    for _ in range(num_test_samples):
        x, y = test_dataset.get_batch()
        predictions = detector.predict(x)
        # TODO: use mAP once implemented
        img_num_tps, img_num_fps, img_num_fns = _object_detection_get_tp_fp_fn(
            y, predictions[0], score_threshold=0.5
        )
        tp_count += img_num_tps
        fp_count += img_num_fps
        fn_count += img_num_fns

    if tp_count + fp_count > 0:
        precision = tp_count / (tp_count + fp_count)
    else:
        precision = 0

    if tp_count + fn_count > 0:
        recall = tp_count / (tp_count + fn_count)
    else:
        recall = 0

    assert precision > 0.78
    assert recall > 0.62
