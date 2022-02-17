from importlib import import_module
import pytest
from armory.data.utils import maybe_download_weights_from_s3
import numpy as np


@pytest.mark.usefixtures("ensure_armory_dirs")
def get_armory_module_and_fn(module_name, fn_attr_name, weights_path):
    module = import_module(module_name)
    fn = getattr(module, fn_attr_name)
    if weights_path is not None:
        weights_path = maybe_download_weights_from_s3(weights_path)
    classifier = fn(model_kwargs={}, wrapper_kwargs={}, weights_path=weights_path)
    return module, fn, classifier


def calculate_accuracy(test_ds, classifier):
    accuracy = 0
    for _ in range(test_ds.batches_per_epoch):
        x, y = test_ds.get_batch()
        predictions = classifier.predict(x)
        accuracy += np.sum(np.argmax(predictions, axis=1) == y) / len(y)
    return accuracy / test_ds.batches_per_epoch


@pytest.mark.parametrize(
    "module_name, fn_attr_name, weights_path, expected_accuracy",
    [
        ("armory.baseline_models.pytorch.mnist", "get_art_model", None, 0.9),
        (
            "armory.baseline_models.pytorch.mnist",
            "get_art_model",
            "undefended_mnist_5epochs.pth",
            0.98,
        ),
    ],
)
def test_model_creation(
    module_name,
    fn_attr_name,
    weights_path,
    expected_accuracy,
    dataset_generator,
    armory_dataset_dir,
):
    module, fn, classifier = get_armory_module_and_fn(
        module_name, fn_attr_name, weights_path
    )

    if weights_path is None:
        train_ds = dataset_generator(
            name="mnist",
            batch_size=600,
            num_epochs=1,
            split="train",
            framework="numpy",
            dataset_dir=armory_dataset_dir,
        )
        classifier.fit_generator(
            train_ds, nb_epochs=1,
        )

    test_ds = dataset_generator(
        name="mnist",
        batch_size=100,
        num_epochs=1,
        split="test",
        framework="numpy",
        dataset_dir=armory_dataset_dir,
    )

    accuracy = calculate_accuracy(test_ds, classifier)
    assert accuracy > expected_accuracy

    # accuracy = 0
    # for _ in range(test_ds.batches_per_epoch):
    #     x, y = test_ds.get_batch()
    #     predictions = classifier.predict(x)
    #     accuracy += np.sum(np.argmax(predictions, axis=1) == y) / len(y)
    # assert (accuracy / test_ds.batches_per_epoch) > 0.9
