"""
Docker orchestration managers for ARMORY.
"""

import logging
import os

import docker

from armory import paths


logger = logging.getLogger(__name__)


class ArmoryInstance(object):
    """
    This object will control a specific docker container.
    """

    def __init__(
        self, image_name, runtime: str = "runc", envs: dict = None, ports: dict = None
    ):
        self.docker_client = docker.from_env()

        os.makedirs(paths.OUTPUTS, exist_ok=True)
        container_args = {
            "runtime": runtime,
            "remove": True,
            "detach": True,
            "volumes": {
                paths.CWD: {"bind": "/workspace", "mode": "rw"},
                paths.USER_ARMORY: {"bind": "/root/.armory", "mode": "rw"},
                paths.USER_KERAS: {"bind": "/root/.keras", "mode": "rw"},
            },
        }
        if ports is not None:
            container_args["ports"] = ports

        if envs:
            container_args["environment"] = envs

        self.docker_container = self.docker_client.containers.run(
            image_name, **container_args
        )

        logger.info(f"ARMORY Instance {self.docker_container.short_id} created.")

    def exec_cmd(self, cmd: str, user=""):
        log = self.docker_container.exec_run(
            cmd, stdout=True, stderr=True, stream=True, user=user
        )

        for out in log.output:
            print(out.decode())

    def __del__(self):
        if hasattr(self, "docker_container"):  # needed if there is an error in __init__
            self.docker_container.stop()


class ManagementInstance(object):
    """
    This object will manage ArmoryInstance objects.
    """

    def __init__(self, image_name: str, runtime="runc"):
        self.instances = {}
        self.runtime = runtime
        self.name = image_name

    def start_armory_instance(
        self, envs: dict = None, ports: dict = None
    ) -> ArmoryInstance:
        temp_inst = ArmoryInstance(
            self.name, runtime=self.runtime, envs=envs, ports=ports
        )
        self.instances[temp_inst.docker_container.short_id] = temp_inst
        return temp_inst

    def stop_armory_instance(self, instance: ArmoryInstance) -> None:
        logger.info(f"Stopping instance: {instance.docker_container.short_id}")
        del self.instances[instance.docker_container.short_id]
