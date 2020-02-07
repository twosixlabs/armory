"""
python -m armory run <json_config>
OR
armory run <json_config>

Try:
    <json_config> = 'examples/mnist_fgm_all_epsilon.json'

This runs an arbitrary config file. Results are output to the `outputs/` directory.
"""

import argparse
import json
import logging
import sys

try:
    import coloredlogs

    from armory.eval import Evaluator
    from armory.docker.management import ManagementInstance
except ImportError as e:
    module = e.name
    print(f"ERROR: cannot import '{module}'", file=sys.stderr)
    try:
        with open("requirements.txt") as f:
            requirements = f.read().splitlines()
    except OSError:
        print(f"ERROR: cannot locate 'requirements.txt'", file=sys.stderr)
        sys.exit(1)

    if module in requirements:
        print(f"    Please run: $ pip install -r requirements.txt", file=sys.stderr)
    else:
        print(f"ERROR: {module} not in requirements. Please submit bug report!!!")
    sys.exit(1)


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
        help="Whether to all interactive access to container",
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

    class PortNumber(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            print((self.metavar,))
            if not 0 < values < 2 ** 16:
                raise argparse.ArgumentError(self, "port numbers must be in (0, 65535]")
            setattr(namespace, self.dest, values)

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
    with open(args.filepath) as f:
        config = json.load(f)
    rig = Evaluator(config)
    if args.interactive or args.jupyter:
        rig.run_interactive()
    else:
        rig.run_config()


def download_all_datasets(command_args, prog, description):
    """
    Script to download all datasets and docker container for offline usage.
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
    args = parser.parse_args(command_args)
    coloredlogs.install(level=args.log_level)

    manager = ManagementInstance(image_name="twosixarmory/tf1:0.2.0")
    runner = manager.start_armory_instance()
    cmd = "; ".join(
        [
            "import logging",
            "import coloredlogs",
            "coloredlogs.install(logging.INFO)",
            "from armory.data import data",
            "data.download_all()",
        ]
    )
    runner.exec_cmd(f"python -c '{cmd}'")
    manager.stop_armory_instance(runner)


# command, (function, description)
PROGRAM = "armory"
COMMANDS = {
    "run": (run, "run armory from config file"),
    "download": (download_all_datasets, "download all datasets used by armory"),
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
            "",
            f"Run '{PROGRAM} <command> --help' for more information on a command.",
            " ",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(usage())
        sys.exit(1)

    parser = argparse.ArgumentParser(prog="armory", usage=usage())

    class Command(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            if values not in COMMANDS:
                raise argparse.ArgumentError(
                    self,
                    f"<command> {values} invalid.\n"
                    f"<command> must be one of {list(COMMANDS)}",
                )
            setattr(namespace, self.dest, values)

    parser.add_argument(
        "command", metavar="<command>", type=str, help="armory command", action=Command,
    )
    args = parser.parse_args(sys.argv[1:2])

    func, description = COMMANDS[args.command]
    prog = f"{PROGRAM} {args.command}"
    func(sys.argv[2:], prog, description)
