import logging
import subprocess

logger = logging.getLogger(__name__)


class HostArmoryInstance:
    def exec_cmd(self, cmd: str, user=""):
        if user:
            raise ValueError("HostArmoryInstance does not support the user input")
        subprocess.run(cmd, shell=True)


class HostManagementInstance:
    def __init__(self):
        self.instance = HostArmoryInstance()

    def start_armory_instance(
        self, envs: dict = None, ports: dict = None, container_subdir: str = None
    ):
        if envs or ports:
            raise ValueError(f"Arguments envs {envs} ports {ports}")
        return self.instance

    def stop_armory_instance(self, instance):
        pass
