"""

"""
import docker
import os
from pathlib import Path

import logging
import coloredlogs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
coloredlogs.install()


class ManagementInstance(object):
    """
    This object will hold ArmoryInstance objects it manages
    """

    def __init__(self):
        self.instances = {}

    def start_armory_instance(self):
        temp_inst = ArmoryInstance()
        self.instances[temp_inst.docker_container.short_id] = temp_inst
        return temp_inst

    def stop_armory_instance(self, instance):
        logger.critical(self.instances)
        del self.instances[instance.docker_container.short_id]


class ArmoryInstance(object):
    """
    This object will contain the docker container being used,)
    """

    def __init__(self):
        """
        """
        _project_root = Path(__file__).parents[2]
        self.output_dir = _project_root / Path("outputs/")
        os.makedirs(self.output_dir, exist_ok=True)

        # Get a tempdir and stand up a container
        self.docker_client = docker.from_env()
        self.disk_location = _project_root

        self.docker_container = self.docker_client.containers.run(
            "twosixlabs/armory:0.1",
            remove=True,
            detach=True,
            volumes={self.disk_location.name: {"bind": "/armory", "mode": "rw"}},
        )

        logger.info(f"ARMORY Instance {self.docker_container.short_id} created.")

    def __del__(self):
        self.docker_container.stop()
