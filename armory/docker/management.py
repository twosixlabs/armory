"""
Docker orchestration managers for ARMORY.
"""

import logging

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
        command: str = "tail -f /dev/null",
        user: str = "",
    ):
        self.docker_client = docker.from_env(version="auto")

        host_paths = paths.HostPaths()
        docker_paths = paths.DockerPaths()

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
                host_paths.local_git_dir: {
                    "bind": docker_paths.local_git_dir,
                    "mode": "rw",
                },
                host_paths.output_dir: {"bind": docker_paths.output_dir, "mode": "rw"},
                host_paths.saved_model_dir: {
                    "bind": docker_paths.saved_model_dir,
                    "mode": "rw",
                },
                host_paths.tmp_dir: {"bind": docker_paths.tmp_dir, "mode": "rw"},
            },
            "shm_size": "16G",
        }

        if ports is not None:
            container_args["ports"] = ports
        if command is not None:
            container_args["command"] = command
        if user:
            container_args["user"] = user
        if envs:
            container_args["environment"] = envs
        self.docker_container = self.docker_client.containers.run(
            image_name, **container_args
        )

        logger.info(f"ARMORY Instance {self.docker_container.short_id} created.")

    def exec_cmd(self, cmd: str, user=""):
        log = self.docker_container.exec_run(
            cmd, stdout=True, stderr=True, stream=True, user=user,
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
        self, envs: dict = None, ports: dict = None, user: str = "",
    ) -> ArmoryInstance:
        temp_inst = ArmoryInstance(
            self.name, runtime=self.runtime, envs=envs, ports=ports, user=user,
        )
        self.instances[temp_inst.docker_container.short_id] = temp_inst
        return temp_inst

    def stop_armory_instance(self, instance: ArmoryInstance) -> None:
        logger.info(f"Stopping instance: {instance.docker_container.short_id}")
        del self.instances[instance.docker_container.short_id]
