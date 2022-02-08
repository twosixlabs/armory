import pytest
from armory.logs import log
import os
import subprocess


@pytest.mark.parametrize("config, args",[("cifar_short.json",["--check","--no-docker"]),
                                         ("cifar_short.json",["--no-docker"]),
                                         ("carla_short.json", ["--check", "--no-docker"]),
                                         ("cifar_short.json",["--no-docker"]),
                                         ])
def test_run(scenario_configs, config, args):
    cf = os.path.join(scenario_configs, "no_docker",config)
    cmd = ["armory","run",cf] + args
    log.info("Executing: {}".format(cmd))
    result = subprocess.run(cmd)
    log.info("Resulting Return Code: {}".format(result.returncode))
    assert result.returncode == 0

