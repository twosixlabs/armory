from importlib import import_module

import pytest

from armory.data.utils import maybe_download_weights_from_s3

# Mark all tests in this file as `unit`
pytestmark = pytest.mark.unit


@pytest.mark.usefixtures("ensure_armory_dirs")
@pytest.mark.online
def get_armory_module_and_fn(
    module_name, fn_attr_name, weights_path, model_kwargs={}, wrapper_kwargs={}
):
    module = import_module(module_name)
    fn = getattr(module, fn_attr_name)
    if weights_path is not None:
        weights_path = maybe_download_weights_from_s3(weights_path)
    classifier = fn(
        model_kwargs=model_kwargs,
        wrapper_kwargs=wrapper_kwargs,
        weights_path=weights_path,
    )
    return module, fn, classifier


@pytest.mark.parametrize(
    "module_name, fn_attr_name, weights_path",
    [
        (
            "armory.baseline_models.pytorch.mnist",
            "get_art_model",
            None,
        ),
        (
            "armory.baseline_models.pytorch.mnist",
            "get_art_model",
            "undefended_mnist_5epochs.pth",
        ),
        (
            "armory.baseline_models.pytorch.cifar",
            "get_art_model",
            None,
        ),
        (
            "armory.baseline_models.pytorch.micronnet_gtsrb",
            "get_art_model",
            None,
        ),
    ],
)
def test_model_creation(module_name, fn_attr_name, weights_path):
    module, fn, classifier = get_armory_module_and_fn(
        module_name, fn_attr_name, weights_path
    )
