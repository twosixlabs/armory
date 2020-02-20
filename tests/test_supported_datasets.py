"""
Ensure data.SUPPORTED_DATASETS are correct.
"""

import unittest

from armory.data import datasets


class SupportedTest(unittest.TestCase):
    def test_functions(self):
        for name, function in datasets.SUPPORTED_DATASETS.items():
            self.assertTrue(
                callable(function),
                f"function {function} for dataset {name} is not callable",
            )
