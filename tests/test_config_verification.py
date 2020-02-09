import pathlib
import unittest

from armory.utils.configuration import load_config


class ConfigurationTest(unittest.TestCase):
    def test_no_config(self):
        with self.assertRaises(FileNotFoundError):
            load_config("not_a_file.txt")

    def test_no_evaluation(self):
        with self.assertRaisesRegex(ValueError, "Evaluation field must contain"):
            load_config("tests/test_data/missing_eval.json")

    @staticmethod
    def test_all_examples():
        example_dir = pathlib.Path("examples/")

        for json_path in example_dir.glob("*.json"):
            load_config(str(json_path))
