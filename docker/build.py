import sys
import shutil
import argparse
import subprocess

from pathlib import Path


script_dir = Path(__file__).parent
root_dir = script_dir.parent

armory_frameworks = ["pytorch", "pytorch-deepspeech", "tf2"]

# NOTE: Podman is not officially supported, but this enables
#       use as a drop-in replacement for building.
container_platform = "docker" if shutil.which("docker") else "podman"


def cli_parser(argv=sys.argv[1:]):
    parser = argparse.ArgumentParser("Armory Container Build Script")
    arguments = (
        (("-f", "--framework"), dict(
            choices=armory_frameworks + ["all"],
            help="Framework to build",
            required=True,
        )),
        (("-b", "--base-tag"), dict(
            help="Version tag for twosixarmory/armory-base",
            default="latest",
            required=False,
        )),
        (("-nc", "--no-cache"), dict(
            action="store_true",
            help="Do not use docker cache",
        )),
        (("-np", "--no-pull"), dict(
            action="store_true",
            help="Do not pull latest base",
        )),
        (("-n", "--dry-run"), dict(
            action="store_true",
            help="Do not build, only print commands",
        )),
        (("-p", "--platform"), dict(
            choices=["docker", "podman"],
            help="Print verbose output",
            default=container_platform,
            required=False,
        )),
    )
    for args, kwargs in arguments:
        parser.add_argument(*args, **kwargs)
    parser.set_defaults(func=init)
    return parser.parse_args(argv)


def normalize_git_version(git_output: str) -> str:
    """Normalize `git describe` output.
    NOTE: This method is similar to the one found in `setup.py` except this
          will return a valid `docker` tag from the version string. This is
          done by replacing `+` characters with `.` characters.

    EXAMPLE:
        >>> git_version = "1.2.3+build4567abc"
        >>> pip_version = git_version
        >>> docker_version = pip_version.replace("+", ".")
    """
    normalized_version = git_output.strip().lstrip('v')
    normalized_version = [part.lstrip('g') for part in normalized_version.split('-')]
    normalized_version = '.build'.join(normalized_version[0::2])
    return normalized_version


def get_tag_version(git_dir: Path = None) -> str:
    '''Retrieve the version from the most recent git tag
    NOTE: In order to ensure consistent versioning, across Armory packaging and
          containers, parts of this method are similar to the ones found in
          `setup.py` and `armory/utils/version`.
    '''
    # TODO: Eventually... there should be a separate method for retrieving version
    #       information that does not require git.
    if shutil.which('git') is None:
        raise RuntimeError('git is not installed')

    git_describe = subprocess.run(
        ['git', 'describe', '--tags'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    if bool(git_describe.returncode) or not bool(git_describe.stdout):
        raise RuntimeError('Unable to retrieve git tag')
    return normalize_git_version(git_describe.stdout.decode('utf-8'))


def build_worker(framework, version, platform, base_tag, **kwargs):
    '''Builds armory container for a given framework.'''
    dockerfile = script_dir / f"Dockerfile-{framework}"
    build_command = [
        f"{platform}",
        "build",
        "--force-rm",
        "--tag",
        f"twosixarmory/{framework}:{version}",
        "--build-arg",
        f"base_image_tag={base_tag}",
        "--build-arg",
        f"armory_version={version}",
        "--file",
        f"{dockerfile}",
        f"{Path().cwd()}",
    ]
    if kwargs.get('no_cache'):
        build_command.insert(3, "--no-cache")
    if not kwargs.get('no_pull'):
        build_command.insert(3, "--pull")
    if not dockerfile.exists():
        sys.exit(f"ERROR:\tError building {framework}!\n"
                 f"\tDockerfile not found: {dockerfile}\n")
    print(f"EXEC\tPreparing to run:\n"
          f"\t\t{' '.join(build_command)}")
    if not kwargs.get("dry_run"):
        subprocess.run(build_command)


def init(*args, **kwargs):
    '''Kicks off the build process.'''
    frameworks = [kwargs.get('framework', False)]
    if frameworks == ["all"]:
        frameworks = armory_frameworks
    armory_version = get_tag_version()
    print(f"EXEC:\tRetrieved version {armory_version} from `git` tags.")
    print("EXEC:\tCleaning up...")
    for key in ["framework", "func"]:
        del kwargs[key]
    for framework in frameworks:
        print(f"EXEC:\tBuilding {framework} container.")
        build_worker(framework, armory_version, **kwargs)


if __name__ == "__main__":
    # Ensure correct location
    if not (root_dir / "armory").is_dir():
        sys.exit(f"ERROR:\tEnsure this script is ran from the root of the armory repo.\n"
                 "\tEXAMPLE:\n"
                 f"\t\t$ python3 {script_dir / 'build.py'}")

    # Ensure docker/podman is installed
    if not shutil.which(container_platform):
        sys.exit("ERROR:\tCannot find compatible container on the system.\n"
                 "\tAsk your system administrator to install either `docker` or `podman`.")

    # Parse CLI arguments
    arguments = cli_parser()
    arguments.func(**vars(arguments))
