#! /usr/bin/env python3

import argparse
from pathlib import Path
import shutil
import subprocess
import sys

from armory import __version__ as armory_version

script_dir = Path(__file__).parent
root_dir = script_dir.parent

armory_frameworks = ["armory", "pytorch-deepspeech", "yolo"]

# NOTE: Podman is not officially supported, but this enables
#       use as a drop-in replacement for building.
container_platform = "docker" if shutil.which("docker") else "podman"


def cli_parser(argv=sys.argv[1:]):
    parser = argparse.ArgumentParser("build.py")
    arguments = (
        (
            ("-f", "--framework"),
            dict(
                choices=armory_frameworks + ["all"],
                help="Framework to build",
                required=True,
            ),
        ),
        (
            ("-b", "--base-tag"),
            dict(
                help="Version tag for twosixarmory/armory-base",
                default="latest",
                required=False,
            ),
        ),
        (
            ("--no-cache"),
            dict(
                action="store_true",
                help="Do not use docker cache",
            ),
        ),
        (
            ("--no-pull"),
            dict(
                action="store_true",
                help="Do not pull latest base",
            ),
        ),
        (
            ("-n", "--dry-run"),
            dict(
                action="store_true",
                help="Do not build, only print commands",
            ),
        ),
        (
            ("-p", "--platform"),
            dict(
                choices=["docker", "podman"],
                help="Print verbose output",
                default=container_platform,
                required=False,
            ),
        ),
    )
    for args, kwargs in arguments:
        args = args if isinstance(args, tuple) else (args,)
        parser.add_argument(*args, **kwargs)
    parser.set_defaults(func=init)

    if len(argv) == 0 or argv[0] in ("usage", "help"):
        parser.print_help()
        sys.exit(1)

    return parser.parse_args(argv)


def build_worker(framework, version, platform, base_tag, **kwargs):
    """Builds armory container for a given framework."""
    # Note: The replace is used to convert the version to a valid docker tag.
    version = version.replace("+", ".")
    dockerfile = script_dir / f"Dockerfile-{framework}"
    build_command = [
        f"{platform}",
        "build",
        "--force-rm",
        "--tag",
        f"twosixarmory/{framework}:{version}",
        "--build-arg",
        f"base_image_tag={base_tag}",
        "--file",
        f"{dockerfile}",
        f"{Path().cwd()}",
    ]
    if kwargs.get("no_cache"):
        build_command.insert(3, "--no-cache")
    if not kwargs.get("no_pull"):
        build_command.insert(3, "--pull")
    if not dockerfile.exists():
        sys.exit(
            f"ERROR:\tError building {framework}!\n"
            f"\tDockerfile not found: {dockerfile}\n"
        )
    print(f"EXEC\tPreparing to run:\n" f"\t\t{' '.join(build_command)}")
    if not kwargs.get("dry_run"):
        return subprocess.run(build_command).returncode


def init(*args, **kwargs):
    """Kicks off the build process."""
    exit_code = 0  # 0 = success, 1 = failure
    frameworks = [kwargs.get("framework", False)]
    if frameworks == ["all"]:
        frameworks = armory_frameworks

    print(f"EXEC:\tRetrieved version {armory_version}.")
    print("EXEC:\tCleaning up...")
    for key in ["framework", "func"]:
        del kwargs[key]
    for framework in frameworks:
        print(f"EXEC:\tBuilding {framework} container.")
        if status := build_worker(framework, armory_version, **kwargs):
            exit_code = status
    sys.exit(exit_code)


if __name__ == "__main__":
    # Ensure correct location
    if not (root_dir / "armory").is_dir():
        sys.exit(
            f"ERROR:\tEnsure this script is ran from the root of the armory repo.\n"
            "\tEXAMPLE:\n"
            f"\t\t$ python3 {root_dir / 'build.py'}"
        )

    # Ensure docker/podman is installed
    if not shutil.which(container_platform):
        sys.exit(
            "ERROR:\tCannot find compatible container on the system.\n"
            "\tAsk your system administrator to install either `docker` or `podman`."
        )

    # Parse CLI arguments
    arguments = cli_parser()
    arguments.func(**vars(arguments))
