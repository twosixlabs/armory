#!/usr/bin/env false
# -*- coding: utf-8 -*-

"""End-to-end pytest fixtures for testing scenario configs.

Example:
    $ pytest --verbose --show-capture=no tests/end_to_end/test_scenario_runner.py

Todo:
    * Decalare this test as being '--no-docker'. Use pytest mark. [woodall]
"""

import re
import json
import pytest

from pathlib import Path

from armory import paths
from armory.__main__ import run


# Marks all tests in this file as `end_to_end`
pytestmark = pytest.mark.end_to_end

scenario_path = Path("scenario_configs")
host_paths    = paths.runtime_paths()
result_path   = host_paths.output_dir


@pytest.fixture(scope="module", autouse=True)
def setup():
    """Pre-flight checks. Future tests will  not continue if setup fails.

    Raises:
        Exception: If `scenario_path` or `result_path` are missing.
    """
    if not all([Path(d).exists() for d in [scenario_path, result_path]]):
        raise Exception(f"Required application paths do not exist!")
        pytest.exit("Previous tests indicate catastrophic failure.")


@pytest.fixture(scope="module")
def scenarios():
    """Scenario generator.

    Returns:
        func: Generator function that yields a scenario config file.
    """
    def scenario_generator():
        """Yields a scenario config file for each test.

        Yields:
            Path: Path object to the scenario config file.

        Raises:
            AssertionError: If the scenario config file is invalid.

        Example:
            >>> print([scenario in scenario_path])
                [Path(<file>), Path(<file>), ...]
        """
        for scenario in scenario_path.glob("**/*.json"):
            try:
                json.loads(scenario.read_text())
                yield scenario
            except json.JSONDecodeError:
                assert False, f"Invalid JSON file: {scenario}"
                continue
    return scenario_generator


def test_scenarios(capsys, scenarios):
    """Scenario test.

    Args:
        scenarios (Iterator): Generator for scenario config files.
        capsys    (:pytest.CaptureFixture:`str`): PyYest builtin fixture to capture stdin/stdout.
    """

    # TODO: Handle cases where docker/torch error out;e.g. GPU not
    #       available or memory is too low. [woodall]
    skip = [
        'carla_video_tracking.json'
    ]

    def scenario_runner():
        """Runs the scenario config and checks the results.

        Yields:
            str: Application stdout for the targeted scenario config file.

        Raises:
            AssertionError: If there is an error in the application stdout.
        """
        for scenario in scenarios():
            # Skip scenarios that require large resource allocations.
            if scenario.name in skip:
                continue

            with capsys.disabled():
                print(f"\n\n{'=' * 42}")
                print(f"\tTesting: {scenario}")

            # Run the scenario & capture the output.
            armory_command = [scenario.as_posix(), "--check", "--use-gpu"]
            run(armory_command, "armory", None)
            out, err = capsys.readouterr()

            # TODO: Check stderr as well. [woodall]
            yield str(out).strip()

    # Check that the results were written.
    for result in scenario_runner():
        result_stdout = str(result).strip()
        test_results_written = "results output written to" in result_stdout.lower()
        test_results_json    = result_stdout.endswith(".json")
        if not all([test_results_written, test_results_json]):
            assert False, f"Invalid output: {out}"
            continue

        # Ensure the file exists.
        result_file = Path(result_stdout.split(" ")[-1])
        assert result_file.exists(), f"Missing result file: {result_file}"
