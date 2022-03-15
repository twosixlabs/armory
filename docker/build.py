import argparse
import docker
import armory
from pathlib import Path
import sys

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

docker_options = {
    "path": str(script_dir),
    "dockerfile": str(dockerfile),
    "tag": f"twosixarmory/{args.framework}:{armory.__version__}",
    "buildargs": {
        "base_image_tag": args.base_tag,
        "armory_version": armory.__version__,
    },
    "rm": True,
    "forcerm": True,
}

if args.no_cache:
    docker_options["nocache"] = True

client = docker.from_env()
if args.dry_run:
    print(f"dry-run is set, would docker build {docker_options}")
    sys.exit(0)

print("running in", Path.cwd())
print("building docker image", docker_options)
client.images.build(**docker_options)
