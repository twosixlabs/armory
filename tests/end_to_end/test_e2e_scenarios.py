#!/usr/bin/env false
# -*- coding: utf-8 -*-

"""End-to-end pytest fixtures for testing scenario configs.

Example:
    $ pip install -e .[developer,engine,math,datasets,pytorch]
    $ pytest --verbose --show-capture=no tests/end_to_end/test_e2e_scenarios.py
    $ pytest --verbose --show-capture=no tests/end_to_end/test_e2e_scenarios.py --scenario-path scenario_configs/cifar10_baseline.json
    $ clear; pytest --verbose tests/end_to_end/test_e2e_scenarios.py --scenario-path scenario_configs/cifar10_baseline.json --github-ci
"""

from pathlib import Path
import unittest

import pytest

from armory import paths
from armory.__main__ import run

# Marks all tests in this file as `end_to_end`
pytestmark = pytest.mark.end_to_end  # noqa: F821


# NOTE: This is a list of all the scenarios that are currently failing to run based
#       on resource constraints. This list should be updated as new scenarios are
#       added to the repo.
block_list = [
    # Evaluation 1-4
    "so2sat_eo_masked_pgd_undefended.json",
    "so2sat_sar_masked_pgd_defended.json",
    "so2sat_sar_masked_pgd_undefended.json",
    "so2sat_eo_masked_pgd_defended.json",
    # Evaluation 6
    "carla_mot_dpatch_defended.json",  # waiting on #1655
]


@pytest.mark.usefixtures("pass_parameters")  # noqa: F821
class TestScenarios(unittest.TestCase):
    @pytest.fixture(autouse=True)  # noqa: F821
    def capsys(self, capsys):
        self.capsys = capsys

    def test_scenarios(self):
        # Setup Armory paths
        paths.set_mode("host")

        capsys = self.capsys
        trapped_in_ci = getattr(self, "github_ci", False)

        scenario_configs = Path("scenario_configs")
        host_paths = paths.runtime_paths()  # noqa: F841

        if hasattr(self, "scenario_path"):
            scenario_path = [Path(getattr(self, "scenario_path"))]
        else:
            scenario_path = [Path(f) for f in list(scenario_configs.glob("**/*.json"))]

        for scenario in scenario_path:
            if trapped_in_ci and scenario.name in block_list:
                continue

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
