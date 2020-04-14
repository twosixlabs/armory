"""
python -m armory run <json_config>
OR
armory run <json_config>

Try:
    <json_config> = 'examples/fgm_attack_binary_search.json'

This runs an arbitrary config file. Results are output to the `outputs/` directory.
"""

import argparse
import logging
import os
import sys

import coloredlogs
import docker
from docker.errors import ImageNotFound

import armory
from armory import paths
from armory.eval import Evaluator
from armory.docker.management import ManagementInstance
from armory.docker import images
from armory.utils import docker_api


class PortNumber(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if not 0 < values < 2 ** 16:
            raise argparse.ArgumentError(self, "port numbers must be in (0, 65535]")
        setattr(namespace, self.dest, values)


class Command(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if values not in COMMANDS:
            raise argparse.ArgumentError(
                self,
                f"{values} invalid.\n" f"<command> must be one of {list(COMMANDS)}",
            )
        setattr(namespace, self.dest, values)


DOCKER_IMAGES = {"tf1": images.TF1, "tf2": images.TF2, "pytorch": images.PYTORCH}


class DockerImage(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if values in images.ALL:
            setattr(namespace, self.dest, values)
        elif values.lower() in DOCKER_IMAGES:
            setattr(namespace, self.dest, DOCKER_IMAGES[values])
        else:
            raise argparse.ArgumentError(
                self,
                f"{values} invalid.\n"
                f" must be one of {DOCKER_IMAGES} or {images.ALL}",
            )


DEFAULT_SCENARIO = "scenarios-set1"


class DownloadConfig(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if values.lower().endswith(".json"):
            config_path = values
        else:
            config_path = os.path.join(
                "armory", "scenarios", "download_configs", values + ".json"
            )

        if os.path.isfile(config_path):
            setattr(namespace, self.dest, config_path)
        else:
            raise argparse.ArgumentError(
                self,
                f"Config json file: {values} not found. Must be {DEFAULT_SCENARIO} or other valid config file path",
            )


def run(command_args, prog, description):
    parser = argparse.ArgumentParser(prog=prog, description=description)
    parser.add_argument(
        "filepath", metavar="<json_config>", type=str, help="json config file"
    )
    parser.add_argument(
        "-d",
        "--debug",
        dest="log_level",
        action="store_const",
        const=logging.DEBUG,
        default=logging.INFO,
        help="Debug output (logging=DEBUG)",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        dest="interactive",
        action="store_const",
        const=True,
        default=False,
        help="Whether to allow interactive access to container",
    )
    parser.add_argument(
        "-j",
        "--jupyter",
        dest="jupyter",
        action="store_const",
        const=True,
        default=False,
        help="Whether to set up Jupyter notebook from container",
    )
    parser.add_argument(
        "-p",
        "--port",
        dest="port",
        type=int,
        action=PortNumber,
        metavar="",
        default=8888,
        help="Port number {0, ..., 65535} to connect to Jupyter on",
    )
    args = parser.parse_args(command_args)

    coloredlogs.install(level=args.log_level)
    paths.host()
    rig = Evaluator(args.filepath)
    rig.run(interactive=args.interactive, jupyter=args.jupyter, host_port=args.port)


def _pull_docker_images(docker_client=None):
    if docker_client is None:
        docker_client = docker.from_env(version="auto")
    for image in images.ALL:
        try:
            docker_client.images.get(image)
        except ImageNotFound:
            if armory.is_dev():
                raise ValueError(
                    "For '-dev', please run 'docker/build-dev.sh' locally before running armory"
                )
            print(f"Image {image} was not found. Downloading...")
            docker_api.pull_verbose(docker_client, image)


def download(command_args, prog, description):
    """
    Script to download all datasets and model weights for offline usage.
    """
    parser = argparse.ArgumentParser(prog=prog, description=description)
    parser.add_argument(
        "-d",
        "--debug",
        dest="log_level",
        action="store_const",
        const=logging.DEBUG,
        default=logging.INFO,
        help="Debug output (logging=DEBUG)",
    )
    parser.add_argument(
        metavar="<download data config file>",
        dest="download_config",
        type=str,
        action=DownloadConfig,
        help="Configuration for download of data, e.g. scenarios-set1",
    )

    parser.add_argument(
        metavar="<scenario>",
        dest="scenario",
        type=str,
        default="all",
        help="scenario for which to download data, 'list' for available scenarios, or blank to download all scenarios",
        nargs="?",
    )

    try:
        args = parser.parse_args(command_args)
    except SystemExit:
        parser.print_help()
        raise

    coloredlogs.install(level=args.log_level)
    paths.host()

    if not armory.is_dev():
        print("Downloading all docker images....")
        _pull_docker_images()

    print("Downloading requested datasets and model weights...")
    manager = ManagementInstance(image_name=images.TF1)
    runner = manager.start_armory_instance()

    cmd = "; ".join(
        [
            "import logging",
            "import coloredlogs",
            "coloredlogs.install(logging.INFO)",
            "from armory.data import datasets",
            "from armory.data import model_weights",
            f'datasets.download_all("{args.download_config}", "{args.scenario}")',
            f'model_weights.download_all("{args.download_config}", "{args.scenario}")',
        ]
    )
    runner.exec_cmd(f"python -c '{cmd}'")
    manager.stop_armory_instance(runner)


def clean(command_args, prog, description):
    parser = argparse.ArgumentParser(prog=prog, description=description)
    parser.add_argument(
        "-d",
        "--debug",
        dest="log_level",
        action="store_const",
        const=logging.DEBUG,
        default=logging.INFO,
        help="Debug output (logging=DEBUG)",
    )
    parser.add_argument(
        "-f",
        "--force",
        dest="force",
        action="store_const",
        const=True,
        default=False,
        help="Whether to remove images of running containers",
    )
    parser.add_argument(
        "--no-download",
        dest="download",
        action="store_const",
        const=False,
        default=True,
        help="If set, will not attempt to pull images before removing existing",
    )
    args = parser.parse_args(command_args)

    coloredlogs.install(level=args.log_level)

    docker_client = docker.from_env(version="auto")
    if args.download:
        print("Pulling the latest docker images")
        _pull_docker_images(docker_client)

    print("Deleting old docker images")
    tags = set()
    for image in docker_client.images.list():
        tags.update(image.tags)

    # If dev version, only remove old dev-tagged containers
    for tag in sorted(tags):
        if images.is_old(tag):
            print(f"Attempting to remove tag {tag}")
            try:
                docker_client.images.remove(tag, force=args.force)
                print(f"* Tag {tag} removed")
            except docker.errors.APIError as e:
                if not args.force and "(must force)" in str(e):
                    print(e)
                    print(f"Cannot delete tag {tag}. Must use `--force`")
                else:
                    raise


def _get_path(name, default_path, absolute_required=True):
    answer = None
    while answer is None:
        try:
            answer = input(f'{name} [DEFAULT: "{default_path}"]: ')
        except EOFError:
            answer = ""
        if not answer:
            answer = default_path
        answer = os.path.expanduser(answer)
        if os.path.isabs(answer):
            answer = os.path.abspath(answer)
        elif absolute_required:
            print(f"Invalid answer: '{answer}' Absolute path required for {name}")
            answer = None
        else:
            answer = os.path.relpath(answer)
    return answer


def configure(command_args, prog, description):
    parser = argparse.ArgumentParser(prog=prog, description=description)
    parser.add_argument(
        "-d",
        "--debug",
        dest="log_level",
        action="store_const",
        const=logging.DEBUG,
        default=logging.INFO,
        help="Debug output (logging=DEBUG)",
    )
    args = parser.parse_args(command_args)
    coloredlogs.install(level=args.log_level)

    default = paths.default()

    instructions = "\n".join(
        [
            "Configuring paths for armory usage",
            f'    This configuration will be stored at "{default.armory_config}"',
            "",
            "Please enter desired target directory for the following paths.",
            "    If left empty, the default path will be used.",
            "    Absolute paths (which include '~' user paths) are required.",
            "",
        ]
    )
    print(instructions)

    config = {
        "dataset_dir": _get_path("dataset_dir", default.dataset_dir),
        "saved_model_dir": _get_path("saved_model_dir", default.saved_model_dir),
        "tmp_dir": _get_path("tmp_dir", default.tmp_dir),
        "output_dir": _get_path("output_dir", default.output_dir),
    }
    resolved = "\n".join(
        [
            "Resolved paths:",
            f"    dataset_dir:     {config['dataset_dir']}",
            f"    saved_model_dir: {config['saved_model_dir']}",
            f"    tmp_dir:         {config['tmp_dir']}",
            f"    output_dir:      {config['output_dir']}",
            "",
        ]
    )
    print(resolved)
    save = None
    while save is None:
        if os.path.isfile(default.armory_config):
            print("WARNING: this will overwrite existing configuration.")
            print("    Press Ctrl-C to abort.")
        answer = input("Save this configuration? [Y/n] ")
        if answer in ("Y", "y", ""):
            print("Saving configuration...")
            paths.save_config(config)
            print("Configure successful")
            save = True
        elif answer in ("N", "n"):
            print("Configuration not saved")
            save = False
        else:
            print(f"Invalid selection: {answer}")
        print()
    print("Configure complete")


def launch(command_args, prog, description):
    parser = argparse.ArgumentParser(prog=prog, description=description)
    parser.add_argument(
        "docker_image",
        metavar="<docker image>",
        type=str,
        help="docker image framework: 'tf1', 'tf2', or 'pytorch'",
        action=DockerImage,
    )
    parser.add_argument(
        "-d",
        "--debug",
        dest="log_level",
        action="store_const",
        const=logging.DEBUG,
        default=logging.INFO,
        help="Debug output (logging=DEBUG)",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        dest="interactive",
        action="store_const",
        const=True,
        default=False,
        help="Whether to allow interactive access to container",
    )
    parser.add_argument(
        "-j",
        "--jupyter",
        dest="jupyter",
        action="store_const",
        const=True,
        default=False,
        help="Whether to set up Jupyter notebook from container",
    )
    parser.add_argument(
        "-p",
        "--port",
        dest="port",
        type=int,
        action=PortNumber,
        metavar="",
        default=8888,
        help="Port number {0, ..., 65535} to connect to Jupyter on",
    )
    parser.add_argument(
        "--use-gpu",
        dest="use_gpu",
        action="store_const",
        const=True,
        default=False,
        help="Whether to use GPU when launching",
    )
    args = parser.parse_args(command_args)

    coloredlogs.install(level=args.log_level)
    paths.host()

    config = {
        "sysconfig": {"use_gpu": args.use_gpu, "docker_image": args.docker_image,}
    }
    rig = Evaluator(config)
    rig.run(interactive=args.interactive, jupyter=args.jupyter, host_port=args.port)


def exec(command_args, prog, description):
    delimiter = "--"
    usage = "armory exec <docker image> [-d] [--use-gpu] -- <exec command>"
    parser = argparse.ArgumentParser(prog=prog, description=description, usage=usage)
    parser.add_argument(
        "docker_image",
        metavar="<docker image>",
        type=str,
        help="docker image framework: 'tf1', 'tf2', or 'pytorch'",
        action=DockerImage,
    )
    parser.add_argument(
        "-d",
        "--debug",
        dest="log_level",
        action="store_const",
        const=logging.DEBUG,
        default=logging.INFO,
        help="Debug output (logging=DEBUG)",
    )
    parser.add_argument(
        "--use-gpu",
        dest="use_gpu",
        action="store_const",
        const=True,
        default=False,
        help="Whether to use GPU when launching",
    )
    try:
        index = command_args.index(delimiter)
    except ValueError:
        print(f"ERROR: delimiter '{delimiter}' is required.")
        parser.print_help()
        sys.exit(1)
    exec_args = command_args[index + 1 :]
    armory_args = command_args[:index]
    if not exec_args:
        print("ERROR: exec command required")
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args(armory_args)

    coloredlogs.install(level=args.log_level)
    paths.host()

    config = {
        "sysconfig": {"use_gpu": args.use_gpu, "docker_image": args.docker_image,}
    }
    rig = Evaluator(config)
    rig.run(command=" ".join(exec_args))


# command, (function, description)
PROGRAM = "armory"
COMMANDS = {
    "run": (run, "run armory from config file"),
    "download": (
        download,
        "download datasets and model weights used for a given evaluation scenario",
    ),
    "clean": (clean, "download new and remove all old armory docker images"),
    "configure": (configure, "set up armory and dataset paths"),
    "launch": (launch, "launch a given docker container in armory"),
    "exec": (exec, "run a single exec command in the container"),
}


def usage():
    lines = [
        f"{PROGRAM} <command>",
        "",
        "ARMORY Adversarial Robustness Evaluation Test Bed",
        "https://github.com/twosixlabs/armory",
        "",
        "Commands:",
    ]
    for name, (func, description) in COMMANDS.items():
        lines.append(f"    {name} - {description}")
    lines.extend(
        [
            "    -v, --version - get current armory version",
            "",
            f"Run '{PROGRAM} <command> --help' for more information on a command.",
            " ",
        ]
    )
    return "\n".join(lines)


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        print(usage())
        sys.exit(1)
    elif sys.argv[1] in ("-v", "--version", "version"):
        print(f"{armory.__version__}")
        sys.exit(0)

    parser = argparse.ArgumentParser(prog="armory", usage=usage())
    parser.add_argument(
        "command", metavar="<command>", type=str, help="armory command", action=Command,
    )
    args = parser.parse_args(sys.argv[1:2])

    func, description = COMMANDS[args.command]
    prog = f"{PROGRAM} {args.command}"
    func(sys.argv[2:], prog, description)


if __name__ == "__main__":
    main()
