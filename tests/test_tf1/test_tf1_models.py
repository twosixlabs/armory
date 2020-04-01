import unittest
import pytest


@pytest.mark.usefixtures("ensure_armory_dirs")
class TF1ModelsTest(unittest.TestCase):
    def test_tf1_mnist(self):
        pass
