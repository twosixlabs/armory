#!/usr/bin/env false
# -*- coding: utf-8 -*-

"""End-to-end pytest fixtures for testing scenario results.

Example:
    $ pytest --verbose --show-capture=no tests/end_to_end/test_check_results.py

Todo:
    * Decalare this test as being '--no-docker'. Use pytest mark. [woodall]
"""

import re
import json
import pprint
import pytest

from pathlib import Path

from armory import paths
from armory.__main__ import run


# Marks all tests in this file as `end_to_end`
pytestmark = pytest.mark.end_to_end

host_paths    = paths.runtime_paths()
result_path   = Path(host_paths.output_dir)

pp = pprint.PrettyPrinter(indent=4, width=80, compact=True)


@pytest.fixture(scope="module", autouse=True)
def setup():
    """Pre-flight checks. Future tests will  not continue if setup fails.

    Raises:
        Exception: If `scenario_path` or `result_path` are missing.
    """
    if not result_path.exists():
        raise Exception(f"Required application paths do not exist!")
        pytest.exit("Previous tests indicate catastrophic failure.")


@pytest.fixture(scope="module")
def results():
    """Results generator.

    Returns:
        func: Generator function that yields a results json file.
    """
    def results_generator():
        """Loops over the results from the user's armory directory.

        Yields:
            dict: Result from the user's armory directory.

        Raises:
            AssertionError: If the result file is invalid.

        Example:
            >>> print([result for result in results_generator()])
                [{<result>}, {<result>}, ...]
        """
        for result in result_path.glob("**/*.json"):
            try:
                yield json.loads(result.read_text())
            except json.JSONDecodeError:
                assert False, f"Invalid JSON file: {result}"
                continue
    return results_generator


def test_scenarios(capsys, results):
    """Result testing.

    Args:
        results (Iterator): Generator for results files.
        capsys  (:pytest.CaptureFixture:`str`): PyYest builtin to capture stdin/stdout.
    """

    for result in results():
        try:
            _config     = result['config']
            config      = _config['sysconfig']['filepath']
            scenario    = config.split("/")[-1].split(".")[0].replace("_", " ").title()
            description = _config['_description']
            dataset     = _config['dataset']['name']
            gpu_used    = _config['sysconfig']['use_gpu']
            results     = result['results']

            with capsys.disabled():
                print("\n".join([
                        f"\n\n{'=' * 80}",
                        f"Scenario:\t{scenario}",
                        f"Description:\t{description}",
                        f"Config:\t\t{config}",
                        f"Dataset:\t{dataset}",
                        f"GPU Used:\t{gpu_used}",
                        f"{'-' * 80}",
                        f"Results:",
                        pp.pformat(results),
                        f"{'=' * 80}"
                    ]))

        except Exception as e:
            assert False, f"Invalid result file: {result}"