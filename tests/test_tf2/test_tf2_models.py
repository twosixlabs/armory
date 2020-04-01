import unittest
import pytest


@pytest.mark.usefixtures("ensure_armory_dirs")
class TF2ModelsTest(unittest.TestCase):
    def test_tf2_mnist(self):
        pass
