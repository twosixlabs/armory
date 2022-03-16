import argparse
import subprocess
from pathlib import Path
import os

import armory

print(f"armory docker builder version {armory.__version__}")
script_dir = Path(__file__).parent

parser = argparse.ArgumentParser(description="builds a docker image for armory")
parser.add_argument(
    "-b", "--base-tag", help="version tag for twosixarmory", default="latest"
)
parser.add_argument(
    "-n", "--dry-run", action="store_true", help="show what would be done"
)
parser.add_argument("--no-cache", action="store_true", help="do not use docker cache")
parser.add_argument("-t", "--tag", help="additional tag for the docker image")
parser.add_argument(
    "framework", help="framework to build (tf2, pytorch, pyrtorch-deepspeech)",
)
args = parser.parse_args()

if args.framework not in ("tf2", "pytorch", "pyrtorch-deepspeech"):
    raise ValueError(f"unknown framework {args.framework}")

dockerfile = script_dir / f"Dockerfile-{args.framework}"
if not dockerfile.exists():
    print("make sure you run this script from the root of the armory repo")
    raise ValueError(f"Dockerfile not found: {dockerfile}")

# TODO: might want pull:True here to get the latest version of the base image

cmd = [
    "docker",
    "build",
    "--file",
    str(dockerfile),
    "--tag",
    f"twosixarmory/{args.framework}:{armory.__version__}",
    "--build-arg",
    f"base_image_tag={args.base_tag}",
    "--build-arg",
    f"armory_version={armory.__version__}",
    "--force-rm",
]
if args.no_cache:
    cmd.append("--no-cache")

cmd.append(os.getcwd())


print("about to run: ", " ".join(cmd))
if args.dry_run:
    print("dry-run requested, not executing build")
else:
    subprocess.run(cmd)
