import logging
import subprocess
import os

logger = logging.getLogger(__name__)


class HostArmoryInstance:
    def __init__(self, envs: dict = None):
        self.env = os.environ
        for k, v in envs.items():
            self.env[k] = v

    def exec_cmd(self, cmd: str, user=""):
        if user:
            raise ValueError("HostArmoryInstance does not support the user input")
        completion = subprocess.run(cmd, env=self.env, shell=True)
        if completion.returncode:
            logger.error("Command did not finish cleanly")
        else:
            logger.info("Command exited cleanly")
        return completion.returncode


class HostManagementInstance:
    def start_armory_instance(
        self, envs: dict = None, ports: dict = None, container_subdir: str = None
    ):
        if ports:
            raise ValueError(f"Arguments ports {ports} not expected!")

        self.instance = HostArmoryInstance(envs=envs)

        return self.instance

    def stop_armory_instance(self, instance):
        pass
