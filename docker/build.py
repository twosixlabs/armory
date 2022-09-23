
import os
import sys
import shutil
import argparse
import subprocess

from pathlib import Path


script_dir = Path(__file__).parent
root_dir   = script_dir.parent

armory_frameworks  = ["pytorch", "pytorch-deepspeech", "tf2"]
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
        (("-v", "--verbose"), dict(
            action = "store_true",
            help   = "Print verbose output",
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


def move_file(src, dest, name=False):
    if dest.is_dir():
        dest_abs  = dest.absolute()
        dest_name = src.name if not name else name
        src.rename(dest_abs / dest_name)
        return Path(dest_abs / dest_name)
    return False


def rm_tree(pth):
    pth = Path(pth)
    if not pth.exists(): return
    for child in pth.glob('*'):
        if child.is_file():
            child.unlink()
        else:
            rm_tree(child)
    pth.rmdir()


def package_worker():
    '''Builds armory sdist & wheel.
    '''
    dist_dir   = Path(root_dir  / "dist")
    if dist_dir.is_dir(): rm_tree(dist_dir) # Cleanup old builds
    subprocess.run(["hatch", "build", "--clean"])
    package = [f for f in dist_dir.iterdir() if f.name.startswith("armory")][0]
    return ".".join(str(package.stem).split('-')[1].split('.')[:3])


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

    # Build armory pip packages & retrieve the version
    # based on pip package naming scheme.
    print(f"EXEC:\tBundling armory python packages.")
    armory_version = package_worker()

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

    # Parse CLI arguments
    arguments = cli_parser()
    arguments.func(**vars(arguments))
