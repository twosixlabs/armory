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
        self,
        image_name,
        runtime: str = "runc",
        envs: dict = None,
        ports: dict = None,
        container_subdir: str = None,
    ):
        self.docker_client = docker.from_env(version="auto")

        host_paths = paths.host()
        docker_paths = paths.docker()
        host_tmp_dir = host_paths.tmp_dir
        host_output_dir = host_paths.output_dir
        if container_subdir:
            host_tmp_dir = os.path.join(host_tmp_dir, container_subdir)
            host_output_dir = os.path.join(host_output_dir, container_subdir)

        container_args = {
            "runtime": runtime,
            "remove": True,
            "detach": True,
            "volumes": {
                host_paths.cwd: {"bind": docker_paths.cwd, "mode": "rw"},
                host_paths.dataset_dir: {
                    "bind": docker_paths.dataset_dir,
                    "mode": "rw",
                },
                host_paths.saved_model_dir: {
                    "bind": docker_paths.saved_model_dir,
                    "mode": "rw",
                },
                host_output_dir: {"bind": docker_paths.output_dir, "mode": "rw"},
                host_tmp_dir: {"bind": docker_paths.tmp_dir, "mode": "rw"},
            },
        }
        if ports is not None:
            container_args["ports"] = ports

        # Windows docker does not require syncronizing file and
        # directoriy permissions via uid and gid.
        if os.name != "nt":
            user_id = os.getuid()
            group_id = os.getgid()
            container_args["user"] = f"{user_id}:{group_id}"

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
        # Needed if there is an error in __init__
        if hasattr(self, "docker_container"):
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
        self, envs: dict = None, ports: dict = None, container_subdir: str = None,
    ) -> ArmoryInstance:
        temp_inst = ArmoryInstance(
            self.name,
            runtime=self.runtime,
            envs=envs,
            ports=ports,
            container_subdir=container_subdir,
        )
        self.instances[temp_inst.docker_container.short_id] = temp_inst
        return temp_inst

    def stop_armory_instance(self, instance: ArmoryInstance) -> None:
        logger.info(f"Stopping instance: {instance.docker_container.short_id}")
        del self.instances[instance.docker_container.short_id]
