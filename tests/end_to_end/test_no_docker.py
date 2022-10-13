import os
import subprocess

import pytest

from armory.logs import log

# Marks all tests in this file as `end_to_end`
pytestmark = pytest.mark.end_to_end


@pytest.mark.parametrize(
    "config, args",
    [
        ("cifar_short.json", ["--check", "--no-docker"]),
        ("cifar_short.json", ["--no-docker"]),
        ("carla_short.json", ["--check", "--no-docker"]),
        ("carla_short.json", ["--no-docker"]),
    ],
)
def test_run(scenario_configs, config, args):
    log.info("Running Armory scenarios from cmd line")
    cf = os.path.join(scenario_configs, "no_docker", config)
    cmd = ["armory", "run", cf] + args
    log.info("Executing: {}".format(cmd))
    result = subprocess.run(cmd)
    log.info("Resulting Return Code: {}".format(result.returncode))
    assert result.returncode == 0


@pytest.mark.parametrize("config", ["cifar_short.json", "carla_short.json"])
def test_interactive(scenario_configs, config):
    # log.info("Executing Config Dir: {}".format(scenario_configs))
    log.info("Running Armory Scenarios interactive in `--no-docker` mode")
    from armory import paths
    from armory.scenarios.main import get as get_scenario

    log.info("Setting Paths to `host`")
    paths.set_mode("host")

    config = os.path.join(scenario_configs, "no_docker", config)
    log.info("Loading Config: {}".format(config))
    s = get_scenario(config).load()
    log.info("Evaluating Config")
    s.evaluate()
