"""Armory Docker Build Script

This script is designed to simplify the build process
for the various docker images.

For most developer use cases, it is sufficient to run:
    `python docker/build.py derived --no-cache`
which will pull the `base` armory image from dockerhub
and then build the `derived` containers (e.g. pytorch, tf2, etc).

If it is necessary to rebuild the base image, use:
    `python docker/build.py all --no-cache --no-pull`
"""
import argparse
import subprocess
from pathlib import Path
import os

try:
    import armory
except ModuleNotFoundError as e:
    if str(e) == "No module named 'armory'":
        raise ModuleNotFoundError("Ensure armory is pip installed before running")
    raise

FRAMEWORKS = ["pytorch", "pytorch-deepspeech", "tf2"]
SCRIPT_DIR = Path(__file__).parent


def main(framework, repo, tag, base_tag, no_cache, no_pull, dry_run, docker_verbose):
    print(f"armory docker builder version: {armory.__version__}")
    if framework == "all":
        frameworks = ["base"] + FRAMEWORKS
    elif framework == "derived":
        frameworks = FRAMEWORKS
    elif framework in FRAMEWORKS or framework == "base":
        frameworks = [framework]
    else:
        raise ValueError(f"unknown framework {framework}")

    for framework in frameworks:
        dockerfile = SCRIPT_DIR / f"Dockerfile-{framework}"
        tag = tag if framework != "base" else base_tag
        print(f"\nBuilding docker image: {repo}/{framework}:{tag} using {dockerfile}\n")
        if not dockerfile.exists():
            print("make sure you run this script from the root of the armory repo")
            raise ValueError(f"Dockerfile not found: {dockerfile}")

        cmd = [
            "docker",
            "build",
            "--file",
            str(dockerfile),
            "--tag",
            f"{repo}/{framework}:{tag}",
            "--build-arg",
            f"base_image_tag={base_tag}",
            "--build-arg",
            f"armory_version={armory.__version__}",
            "--force-rm",
            f"{docker_verbose}",
        ]
        if no_cache:
            cmd.append("--no-cache")
        if not no_pull:
            cmd.append("--pull")

        cmd.append(os.getcwd())

        print("about to run: ", " ".join(cmd))
        if dry_run:
            print("dry-run requested, not executing build")
        else:
            subprocess.run(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "-b", "--base-tag", help="version tag for twosixarmory", default="latest"
    )
    parser.add_argument(
        "-t",
        "--tag",
        default=armory.__version__,
        help="Tag to apply to derived images (NOTE: for `base` the `--base-tag` will be used",
    )
    parser.add_argument(
        "-r", "--repo", default="twosixarmory", help="Docker Repo to use for image name"
    )
    parser.add_argument(
        "--no-cache", action="store_true", help="do not use docker cache"
    )
    parser.add_argument(
        "--no-pull", action="store_true", help="do not pull latest base"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="show what would be done"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        default="--progress=auto",
        const="--progress-plain",
        action="store_const",
        help="Make Docker build more verbose",
    )
    parser.add_argument(
        "framework",
        choices=["base", "all", "derived"] + FRAMEWORKS,
        help="framework to build.  `derived` specifies all images except base",
    )
    args = parser.parse_args()
    main(
        args.framework,
        args.repo,
        args.tag,
        args.base_tag,
        args.no_cache,
        args.no_pull,
        args.dry_run,
        args.verbose,
    )
