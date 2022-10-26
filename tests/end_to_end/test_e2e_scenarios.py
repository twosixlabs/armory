#!/usr/bin/env false
# -*- coding: utf-8 -*-

"""End-to-end pytest fixtures for testing scenario configs.

Example:
    $ pip install -e .[developer,engine,math,datasets,pytorch]
    $ pytest --verbose --show-capture=no tests/end_to_end/test_scenario_runner.py
    $ pytest --verbose --show-capture=no tests/end_to_end/test_scenario_runner.py --scenario-path scenario_configs/cifar10_baseline.json
    $ clear; pytest --verbose tests/end_to_end/test_scenario_runner.py --scenario-path scenario_configs/cifar10_baseline.json --github-ci
"""

import pytest
import unittest

from pathlib import Path

from armory import paths
from armory.__main__ import run


# Marks all tests in this file as `end_to_end`
pytestmark = pytest.mark.end_to_end  # noqa: F821


# NOTE: This is a list of all the scenarios that are currently failing to run based
#       on resource constraints. This list should be updated as new scenarios are
#       added to the repo.
block_list = [
    "./scenario_configs/eval6/carla_mot/carla_mot_dpatch_defended.json",  # waiting on #1655
]


@pytest.mark.usefixtures("pass_parameters")  # noqa: F821
class TestScenarios(unittest.TestCase):
    @pytest.fixture(autouse=True)  # noqa: F821
    def capsys(self, capsys):
        self.capsys = capsys

    def test_scenarios(self):
        capsys = self.capsys

        trapped_in_ci = getattr(self, "github_ci", False)
        scenario_path = (
            [Path(self.scenario_path)] if hasattr(self, "scenario_path") else []
        )
        scenario_configs = Path("scenario_configs")
        host_paths = paths.runtime_paths()  # noqa: F841

        # Setup Armory paths
        paths.set_mode("host")

        if not len(scenario_path):
            scenario_path = [
                Path(f)
                for f in list(scenario_configs.glob("**/*.json"))
                if f.name not in block_list
            ]

        for scenario in scenario_path:

            if scenario not in block_list:
                try:
                    armory_flags = [
                        scenario.as_posix(),
                        "--no-docker",
                        "--check",
                        "--no-gpu",
                    ]
                    run(armory_flags, "armory", None)
                    out, err = capsys.readouterr()
                except Exception as e:
                    assert False, f"Failed to run scenario: {scenario} - {e}"

                if trapped_in_ci:
                    Path(f"/tmp/.armory/{scenario.name}.log").write_text(
                        "\n\n".join([out, err])
                    )
