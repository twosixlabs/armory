"""
Ensure data.SUPPORTED_DATASETS are correct.
"""
from armory.data import datasets


def test_functions():
    for name, function in datasets.SUPPORTED_DATASETS.items():
        assert callable(function)
