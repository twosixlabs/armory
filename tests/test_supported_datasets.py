"""
Ensure common.SUPPORTED_DATASETS (which doesn't depend on external libs)
   and data.SUPPORTED_DATASETS match.
"""

import unittest

from armory.data import datasets, common


class SupportedTest(unittest.TestCase):
    def test_support(self):
        for k in common.SUPPORTED_DATASETS:
            self.assertIn(
                k,
                datasets.SUPPORTED_DATASETS,
                f"{k} in common.SUPPORTED_DATASETS but not data.SUPPORTED_DATASETS",
            )
        for k in datasets.SUPPORTED_DATASETS:
            self.assertIn(
                k,
                common.SUPPORTED_DATASETS,
                f"{k} in data.SUPPORTED_DATASETS but not common.SUPPORTED_DATASETS",
            )

    def test_functions(self):
        for name, function in datasets.SUPPORTED_DATASETS.items():
            self.assertTrue(
                callable(function),
                f"function {function} for dataset {name} is not callable",
            )
