"""
python -m armory run <json_config>
OR
armory run <json_config>

Try:
    <json_config> = 'examples/fgm_attack_binary_search.json'

This runs an arbitrary config file. Results are output to the `outputs/` directory.
"""

import argparse
import json
import logging
import os
import sys

import coloredlogs
import docker
from docker.errors import ImageNotFound
from jsonschema import ValidationError

import armory
from armory import paths
from armory.configuration import load_global_config, save_config
from armory.eval import Evaluator
from armory.docker import images
from armory.utils import docker_api
from armory.utils.configuration import load_config, load_config_stdin

logger = logging.getLogger(__name__)


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
            print(
                f"WARNING: {values} not in "
                f"{list(DOCKER_IMAGES.keys()) + list(DOCKER_IMAGES.values())}. "
                "Attempting to load custom Docker image."
            )
            setattr(namespace, self.dest, values)


OLD_SCENARIOS = [
    "https://github.com/twosixlabs/armory-example/blob/master/scenario_download_configs/scenarios-set1.json"
]
DEFAULT_SCENARIO = "https://github.com/twosixlabs/armory-example/blob/master/scenario_download_configs/scenarios-set2.json"


class DownloadConfig(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if values.lower().endswith(".json") and os.path.isfile(values):
            setattr(namespace, self.dest, values)
        else:
            raise argparse.ArgumentError(
                self,
                f"Please provide a json config file. See the armory-example repo: "
                f"{DEFAULT_SCENARIO}",
            )


# Helper functions for parsers


def _debug(parser):
    parser.add_argument(
        "-d",
        "--debug",
        dest="log_level",
        action="store_const",
        const=logging.DEBUG,
        default=logging.INFO,
        help="Debug output (logging=DEBUG)",
    )


def _interactive(parser):
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Whether to allow interactive access to container",
    )


def _jupyter(parser):
    parser.add_argument(
        "-j",
        "--jupyter",
        action="store_true",
        help="Whether to set up Jupyter notebook from container",
    )


def _port(parser):
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        action=PortNumber,
        metavar="",
        default=None,
        help=(
            "Port number {0, ..., 65535} to expose from docker container. If --jupyter "
            "flag is set then this port will be used for the jupyter server."
        ),
    )


def _no_gpu(parser):
    parser.add_argument(
        "--no-gpu", action="store_true", help="Whether to not use GPU(s)",
    )


def _use_gpu(parser):
    parser.add_argument(
        "--use-gpu", action="store_true", help="Whether to use GPU(s)",
    )


def _gpus(parser):
    parser.add_argument(
        "--gpus",
        type=str,
        help="Which specific GPU(s) to use, such as '3', '1,5', or 'all'",
    )


def _docker_image(parser):
    parser.add_argument(
        "docker_image",
        metavar="<docker image>",
        type=str,
        help="docker image framework: 'tf1', 'tf2', or 'pytorch'",
        action=DockerImage,
    )


def _docker_image_optional(parser):
    parser.add_argument(
        "--docker-image",
        default=images.TF1,
        metavar="<docker image>",
        type=str,
        help="docker image framework: 'tf1', 'tf2', or 'pytorch'",
        action=DockerImage,
    )


def _skip_docker_images(parser):
    parser.add_argument(
        "--skip-docker-images",
        action="store_true",
        help="Whether to skip downloading docker images",
    )


def _no_docker(parser):
    parser.add_argument(
        "--no-docker",
        action="store_true",
        help="Whether to use Docker or the local host environment",
    )


def _root(parser):
    parser.add_argument(
        "--root", action="store_true", help="Whether to run docker as root",
    )


# Config


def _set_gpus(config, use_gpu, no_gpu, gpus):
    """
    Set gpu values from parser in config
    """
    if (use_gpu or gpus) and no_gpu:
        raise ValueError("no_gpu cannot be set with use_gpu or gpus!")

    if gpus:
        if not use_gpu:
            logger.info("--gpus field specified. Setting --use-gpu to True")
            use_gpu = True
        config["sysconfig"]["gpus"] = gpus

    if use_gpu or "use_gpu" not in config["sysconfig"]:
        # Override if use_gpu, otherwise if config exists, leave config setting in place
        config["sysconfig"]["use_gpu"] = use_gpu
    elif no_gpu:
        config["sysconfig"]["use_gpu"] = False


def _set_outputs(config, output_dir, output_filename):
    if output_dir:
        config["sysconfig"]["output_dir"] = output_dir
    if output_filename:
        config["sysconfig"]["output_filename"] = output_filename


# Commands


def run(command_args, prog, description):
    parser = argparse.ArgumentParser(prog=prog, description=description)
    parser.add_argument(
        "filepath",
        metavar="<json_config>",
        type=str,
        help="json config file. Use '-' to accept standard input or pipe.",
    )
    _debug(parser)
    _interactive(parser)
    _jupyter(parser)
    _port(parser)
    _use_gpu(parser)
    _no_gpu(parser)
    _gpus(parser)
    _no_docker(parser)
    _root(parser)
    parser.add_argument(
        "--output-dir", type=str, help="Override of default output directory prefix",
    )
    parser.add_argument(
        "--output-filename",
        type=str,
        help="Override of default output filename prefix",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Whether to quickly check to see if scenario code runs",
    )
    parser.add_argument(
        "--num-eval-batches",
        type=int,
        help="Number of batches to use for evaluation of benign and adversarial examples",
    )
    parser.add_argument(
        "--skip-benign",
        action="store_true",
        help="Skip benign inference and metric calculations",
    )
    parser.add_argument(
        "--skip-attack",
        action="store_true",
        help="Skip attack generation and metric calculations",
    )
    parser.add_argument(
        "--validate-config",
        action="store_true",
        help="Validate model configuration against several checks",
    )

    args = parser.parse_args(command_args)
    coloredlogs.install(level=args.log_level)

    try:
        if args.filepath == "-":
            if sys.stdin.isatty():
                logging.error(
                    "Cannot read config from raw 'stdin'; must pipe or redirect a file"
                )
                sys.exit(1)
            logger.info("Reading config from stdin...")
            config = load_config_stdin()
        else:
            config = load_config(args.filepath)
    except ValidationError as e:
        logger.error(
            f"Could not validate config: {e.message} @ {'.'.join(e.absolute_path)}"
        )
        sys.exit(1)
    except json.decoder.JSONDecodeError:
        if args.filepath == "-":
            logger.error("'stdin' did not provide a json-parsable input")
        else:
            logger.error(f"Could not decode '{args.filepath}' as a json file.")
            if not args.filepath.lower().endswith(".json"):
                logger.warning(f"{args.filepath} is not a '*.json' file")
        sys.exit(1)
    _set_gpus(config, args.use_gpu, args.no_gpu, args.gpus)
    _set_outputs(config, args.output_dir, args.output_filename)

    rig = Evaluator(config, no_docker=args.no_docker, root=args.root)
    exit_code = rig.run(
        interactive=args.interactive,
        jupyter=args.jupyter,
        host_port=args.port,
        check_run=args.check,
        num_eval_batches=args.num_eval_batches,
        skip_benign=args.skip_benign,
        skip_attack=args.skip_attack,
        validate_config=args.validate_config,
    )
    sys.exit(exit_code)


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
            logger.info(f"Image {image} was not found. Downloading...")
            docker_api.pull_verbose(docker_client, image)


def download(command_args, prog, description):
    """
    Script to download all datasets and model weights for offline usage.
    """
    parser = argparse.ArgumentParser(prog=prog, description=description)
    _debug(parser)
    _docker_image_optional(parser)
    _skip_docker_images(parser)
    parser.add_argument(
        metavar="<download data config file>",
        dest="download_config",
        type=str,
        action=DownloadConfig,
        help=f"Configuration for download of data. See {DEFAULT_SCENARIO}. Note: file must be under current working directory.",
    )
    parser.add_argument(
        metavar="<scenario>",
        dest="scenario",
        type=str,
        default="all",
        help="scenario for which to download data, 'list' for available scenarios, or blank to download all scenarios",
        nargs="?",
    )
    _no_docker(parser)

    args = parser.parse_args(command_args)
    coloredlogs.install(level=args.log_level)

    if args.no_docker:
        logger.info("Downloading requested datasets and model weights in host mode...")
        paths.set_mode("host")
        from armory.data import datasets
        from armory.data import model_weights

        datasets.download_all(args.download_config, args.scenario)
        model_weights.download_all(args.download_config, args.scenario)
        return

    if args.skip_docker_images:
        logger.info("Skipping docker image downloads...")
    elif armory.is_dev():
        logger.info("Dev version. Must build docker images locally with build-dev.sh")
    else:
        logger.info("Downloading all docker images...")
        _pull_docker_images()

    logger.info("Downloading requested datasets and model weights...")
    config = {"sysconfig": {"docker_image": args.docker_image}}

    rig = Evaluator(config)
    cmd = "; ".join(
        [
            "import logging",
            "import coloredlogs",
            f"coloredlogs.install({args.log_level})",
            "from armory.data import datasets",
            "from armory.data import model_weights",
            f'datasets.download_all("{args.download_config}", "{args.scenario}")',
            f'model_weights.download_all("{args.download_config}", "{args.scenario}")',
        ]
    )
    exit_code = rig.run(command=f"python -c '{cmd}'")
    sys.exit(exit_code)


def clean(command_args, prog, description):
    parser = argparse.ArgumentParser(prog=prog, description=description)
    _debug(parser)
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Whether to remove images of running containers",
    )
    parser.add_argument(
        "--no-download",
        dest="download",
        action="store_false",
        help="If set, will not attempt to pull images before removing existing",
    )

    args = parser.parse_args(command_args)
    coloredlogs.install(level=args.log_level)

    docker_client = docker.from_env(version="auto")
    if args.download:
        logger.info("Pulling the latest docker images")
        _pull_docker_images(docker_client)

    logger.info("Deleting old docker images")
    tags = set()
    for image in docker_client.images.list():
        tags.update(image.tags)

    # If dev version, only remove old dev-tagged containers
    for tag in sorted(tags):
        if images.is_old(tag):
            logger.info(f"Attempting to remove tag {tag}")
            try:
                docker_client.images.remove(tag, force=args.force)
                logger.info(f"* Tag {tag} removed")
            except docker.errors.APIError as e:
                if not args.force and "(must force)" in str(e):
                    logger.exception(f"Cannot delete tag {tag}. Must use `--force`")
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


def _get_verify_ssl():
    verify_ssl = None
    while verify_ssl is None:
        answer = input("Verify SSL during downloads? [Y/n] ")
        if answer in ("Y", "y", ""):
            verify_ssl = True
        elif answer in ("N", "n"):
            verify_ssl = False
        else:
            print(f"Invalid selection: {answer}")
        print()
        return verify_ssl


def configure(command_args, prog, description):
    parser = argparse.ArgumentParser(prog=prog, description=description)
    _debug(parser)

    args = parser.parse_args(command_args)
    coloredlogs.install(level=args.log_level)

    default_host_paths = paths.HostDefaultPaths()

    config = None
    if os.path.exists(default_host_paths.armory_config):
        response = None
        while response is None:
            prompt = f"Existing configuration found: {default_host_paths.armory_config}"
            print(prompt)

            response = input("Load existing configuration? [Y/n]")
            if response in ("Y", "y", ""):
                print("Loading configuration...")
                config = load_global_config(
                    default_host_paths.armory_config, validate=False
                )
                print("Load successful")
            elif response in ("N", "n"):
                print("Configuration not loaded")
            else:
                print(f"Invalid selection: {response}")
                response = None
            print()

    instructions = "\n".join(
        [
            "Configuring paths for armory usage",
            f'    This configuration will be stored at "{default_host_paths.armory_config}"',
            "",
            "Please enter desired target directory for the following paths.",
            "    If left empty, the default path will be used.",
            "    Absolute paths (which include '~' user paths) are required.",
            "",
        ]
    )
    print(instructions)

    default_dataset_dir = (
        config["dataset_dir"]
        if config is not None and "dataset_dir" in config.keys()
        else default_host_paths.dataset_dir
    )
    default_local_dir = (
        config["local_git_dir"]
        if config is not None and "local_git_dir" in config.keys()
        else default_host_paths.local_git_dir
    )
    default_saved_model_dir = (
        config["saved_model_dir"]
        if config is not None and "saved_model_dir" in config.keys()
        else default_host_paths.saved_model_dir
    )
    default_tmp_dir = (
        config["tmp_dir"]
        if config is not None and "tmp_dir" in config.keys()
        else default_host_paths.tmp_dir
    )
    default_output_dir = (
        config["output_dir"]
        if config is not None and "output_dir" in config.keys()
        else default_host_paths.output_dir
    )

    config = {
        "dataset_dir": _get_path("dataset_dir", default_dataset_dir),
        "local_git_dir": _get_path("local_git_dir", default_local_dir),
        "saved_model_dir": _get_path("saved_model_dir", default_saved_model_dir),
        "tmp_dir": _get_path("tmp_dir", default_tmp_dir),
        "output_dir": _get_path("output_dir", default_output_dir),
        "verify_ssl": _get_verify_ssl(),
    }
    resolved = "\n".join(
        [
            "Resolved paths:",
            f"    dataset_dir:     {config['dataset_dir']}",
            f"    local_git_dir:   {config['local_git_dir']}",
            f"    saved_model_dir: {config['saved_model_dir']}",
            f"    tmp_dir:         {config['tmp_dir']}",
            f"    output_dir:      {config['output_dir']}",
            "Download options:",
            f"    verify_ssl:      {config['verify_ssl']}",
            "",
        ]
    )
    print(resolved)
    save = None
    while save is None:
        if os.path.isfile(default_host_paths.armory_config):
            print("WARNING: this will overwrite existing configuration.")
            print("    Press Ctrl-C to abort.")
        answer = input("Save this configuration? [Y/n] ")
        if answer in ("Y", "y", ""):
            print("Saving configuration...")
            save_config(config, default_host_paths.armory_dir)
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
    _docker_image(parser)
    _debug(parser)
    _interactive(parser)
    _jupyter(parser)
    _port(parser)
    _use_gpu(parser)
    _no_gpu(parser)
    _gpus(parser)
    _root(parser)

    args = parser.parse_args(command_args)
    coloredlogs.install(level=args.log_level)

    config = {"sysconfig": {"docker_image": args.docker_image}}
    _set_gpus(config, args.use_gpu, args.no_gpu, args.gpus)

    rig = Evaluator(config, root=args.root)
    exit_code = rig.run(
        interactive=args.interactive,
        jupyter=args.jupyter,
        host_port=args.port,
        command="true # No-op",
    )
    sys.exit(exit_code)


def exec(command_args, prog, description):
    delimiter = "--"
    usage = f"armory exec <docker image> [-d] [--use-gpu] {delimiter} <exec command>"
    parser = argparse.ArgumentParser(prog=prog, description=description, usage=usage)
    _docker_image(parser)
    _debug(parser)
    _use_gpu(parser)
    _gpus(parser)
    _no_gpu(parser)
    _root(parser)

    try:
        index = command_args.index(delimiter)
    except ValueError:
        print(f"ERROR: delimiter '{delimiter}' is required.")
        parser.print_help()
        sys.exit(1)
    exec_args = command_args[index + 1 :]
    armory_args = command_args[:index]
    if exec_args:
        command = " ".join(exec_args)
    else:
        print("ERROR: exec command required")
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args(armory_args)
    coloredlogs.install(level=args.log_level)

    config = {"sysconfig": {"docker_image": args.docker_image}}
    # Config
    _set_gpus(config, args.use_gpu, args.no_gpu, args.gpus)

    rig = Evaluator(config, root=args.root)
    exit_code = rig.run(command=command)
    sys.exit(exit_code)


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
