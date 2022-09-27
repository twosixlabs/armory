
import os
import sys
import shutil
import argparse
import subprocess

from pathlib import Path


script_dir = Path(__file__).parent
root_dir   = script_dir.parent

armory_frameworks  = ["pytorch", "pytorch-deepspeech", "tf2"]

# NOTE: Podman is not officially supported, but this enables
#       use as a drop-in replacement for building.
container_platform = "docker" if shutil.which("docker") else "podman"


def cli_parser(argv=sys.argv[1:]):
    parser    = argparse.ArgumentParser("Armory Container Build Script")
    arguments = (
        (("-f", "--framework"), dict(
            choices  = armory_frameworks + ["all"],
            help     = "Framework to build",
            required = True,
        )),
        (("-b", "--base-tag"), dict(
            help     = "Version tag for twosixarmory/armory-base",
            default  = "latest",
            required = False,
        )),
        (("-nc", "--no-cache"), dict(
            action = "store_true",
            help   = "Do not use docker cache",
        )),
        (("-np", "--no-pull"), dict(
            action = "store_true",
            help   = "Do not pull latest base",
        )),
        (("-n", "--dry-run"), dict(
            action = "store_true",
            help   = "Do not build, only print commands",
        )),
        (("-s", "--short-tag"), dict(
            action = "store_true",
            help   = "Use dirty git tag.",
        )),
        (("-p", "--platform"), dict(
            choices  = ["docker", "podman"],
            help     ="Print verbose output",
            default  = container_platform,
            required = False,
        )),
    )
    for args, kwargs in arguments:
        parser.add_argument(*args, **kwargs)
    parser.set_defaults(func=init)
    return parser.parse_args(argv)


def get_version_tag(use_short_tag=False):
    '''Returns the current git tag version.
    '''
    if Path('.git') is None or shutil.which('git') is None:
        sys.exit(f"Unable to find `.git` directory or git is not installed.")

    git_command = ["git", "describe"]

    if use_short_tag:
        git_command.extend(["--tags", "--abbrev=0"])

    git_tag = subprocess.run(
        git_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if git_tag.returncode != 0:
        sys.exit(f"ERROR:\tError retrieving git tag version!\n" \
                 f"\t{git_tag.stderr.decode('utf-8')}")

    return git_tag.stdout.decode('utf-8').strip()[1:]


def build_worker(framework, version, platform, base_tag, **kwargs):
    '''Builds armory container for a given framework.
    '''
    dockerfile    = script_dir / f"Dockerfile-{framework}"
    build_command = [
        f"{platform}",
        f"build",
        f"--force-rm",
        f"--tag",
        f"twosixarmory/{framework}:{version}",
        f"--build-arg",
        f"base_image_tag={base_tag}",
        f"--build-arg",
        f"armory_version={version}",
        f"--file",
        f"{dockerfile}",
        f"{Path().cwd()}",
    ]
    if kwargs.get('no_cache'):    build_command.insert(3, "--no-cache")
    if not kwargs.get('no_pull'): build_command.insert(3, "--pull")

    if not dockerfile.exists():
        sys.exit(f"ERROR:\tError building {framework}!\n" \
                 f"\tDockerfile not found: {dockerfile}\n")

    print(f"EXEC\tPreparing to run:\n"    \
          f"\t\t{' '.join(build_command)}")

    if not kwargs.get("dry_run"):
        subprocess.run(build_command)


def init(*args, **kwargs):
    '''Kicks off the build process.
    '''
    if (frameworks := [kwargs.get('framework')]) == ["all"]:
        frameworks = armory_frameworks

    print(f"EXEC:\tRetrieving version from `git` tags.")
    use_short_tag = kwargs.get('short_tag', False)
    armory_version = get_version_tag(use_short_tag)

    print(f"EXEC:\tCleaning up...")
    for key in ["framework", "func"]: del kwargs[key]

    for framework in frameworks:
        print(f"EXEC:\tBuilding {framework} container.")
        build_worker(framework, armory_version, **kwargs)



if __name__ == "__main__":
    # Ensure correct location
    if not (root_dir / "armory").is_dir():
        sys.exit(f"ERROR:\tEnsure this script is ran from the root of the armory repo.\n" \
                 f"\tEXAMPLE:\n"                                                          \
                 f"\t\t$ python3 {script_dir / 'build.py'}")

    # Ensure docker/podman is installed
    if not shutil.which(container_platform):
        sys.exit(f"ERROR:\tCannot find compatible container on the system.\n" \
                 f"\tAsk your system administrator to install either `docker` or `podman`.")

    # Parse CLI arguments
    arguments = cli_parser()
    arguments.func(**vars(arguments))

