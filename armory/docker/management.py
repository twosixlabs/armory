"""
Docker orchestration managers for ARMORY.
"""

import datetime
import logging
import os
import shutil

import docker

from armory import paths


logger = logging.getLogger(__name__)


def tmp_output_subdir(retries=10):
    """
    Return the name of the tmp and output subdirectories.

    retries - number of times to retry folder creation before returning an error
        if retries < 0, it will retry indefinitely.
    """
    tries = int(retries) + 1
    host_paths = paths.host()
    while tries:
        try:
            subdir = datetime.datetime.utcnow().isoformat()
            tmp_subdir = os.path.join(host_paths.tmp_dir, subdir)
            output_subdir = os.path.join(host_paths.output_dir, subdir)
            os.mkdir(tmp_subdir)
            os.mkdir(output_subdir)
            return tmp_subdir, output_subdir
        except FileExistsError:
            tries -= 1
            if tries:
                logger.warning(
                    f"Failed to create {tmp_subdir} or {output_subdir}. Retrying..."
                )
    raise ValueError("Failed to create tmp and output subdirectories")


class ArmoryInstance(object):
    """
    This object will control a specific docker container.
    """

    def __init__(
        self, image_name, runtime: str = "runc", envs: dict = None, ports: dict = None
    ):
        host_paths = paths.host()
        docker_paths = paths.docker()
        self.docker_client = docker.from_env(version="auto")

        self.tmp_subdir, self.output_subdir = tmp_output_subdir()
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
                self.output_subdir: {"bind": docker_paths.output_dir, "mode": "rw"},
                self.tmp_subdir: {"bind": docker_paths.tmp_dir, "mode": "rw"},
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
        logger.info(f"Deleting tmp_output_subdir {self.tmp_output_subdir}")
        try:
            shutil.rmtree(self.tmp_output_subdir)
        except OSError:
            logger.warning(f"Failed to delete {self.tmp_output_subdir}")
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
