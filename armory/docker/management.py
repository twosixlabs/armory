"""
Docker orchestration managers for ARMORY.
"""


import armory
from armory import paths
from armory.docker import images
from armory.logs import log
import docker


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

        mounts = [
            docker.types.Mount(
                source=getattr(host_paths, dir),
                target=getattr(docker_paths, dir),
                type="bind",
                read_only=False,
            )
            for dir in "cwd dataset_dir local_git_dir output_dir saved_model_dir tmp_dir".split()
        ]

        container_args = {
            "runtime": runtime,
            "remove": True,
            "detach": True,
            "mounts": mounts,
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

        log.info(f"ARMORY Instance {self.docker_container.short_id} created.")

    def exec_cmd(self, cmd: str, user="", expect_sentinel=True) -> int:
        # We would like to check the return code to see if the command ran cleanly,
        #  but `exec_run()` cannot both return the code and stream logs
        # https://docker-py.readthedocs.io/en/stable/containers.html#docker.models.containers.Container.exec_run
        result = self.docker_container.exec_run(
            cmd,
            stdout=True,
            stderr=True,
            stream=True,
            tty=True,
            user=user,
        )

        # the sentinel should be the last output from the container
        # but threading may cause certain warning messages to be printed during container shutdown
        #  ie after the sentinel
        sentinel_found = False
        for out in result.output:
            output = out.decode(encoding="utf-8", errors="replace").strip()
            if not output:  # skip empty lines
                continue
            # this looks absurd, but in some circumstances result.output will combine
            #  outputs from the container into a single string
            # eg, print(a); print(b) is delivered as 'a\r\nb'
            for inner_line in output.splitlines():
                inner_output = inner_line.strip()
                if not inner_output:
                    continue
                print(inner_output)
                if inner_output == armory.END_SENTINEL:
                    sentinel_found = True

        # if we're not running a config (eg armory exec or launch)
        #  we don't expect the sentinel to be printed and we have no way of
        #  knowing if the command ran cleanly so we return unconditionally
        if not expect_sentinel:
            return 0
        if sentinel_found:
            log.success("command exited cleanly")
            return 0
        else:
            log.error(f"command {cmd} did not finish cleanly")
            return 1

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
        self.name = images.ensure_image_present(image_name)

    def start_armory_instance(
        self,
        envs: dict = None,
        ports: dict = None,
        user: str = "",
    ) -> ArmoryInstance:
        temp_inst = ArmoryInstance(
            self.name,
            runtime=self.runtime,
            envs=envs,
            ports=ports,
            user=user,
        )
        self.instances[temp_inst.docker_container.short_id] = temp_inst
        return temp_inst

    def stop_armory_instance(self, instance: ArmoryInstance) -> None:
        log.info(f"Stopping instance: {instance.docker_container.short_id}")
        del self.instances[instance.docker_container.short_id]
