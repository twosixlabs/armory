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

print(f"armory docker builder version {armory.__version__}")
script_dir = Path(__file__).parent

parser = argparse.ArgumentParser(description="builds a docker image for armory")
parser.add_argument(
    "-b", "--base-tag", help="version tag for twosixarmory", default="latest"
)
parser.add_argument("--no-cache", action="store_true", help="do not use docker cache")
parser.add_argument("--no-pull", action="store_true", help="do not pull latest base")
parser.add_argument(
    "-n", "--dry-run", action="store_true", help="show what would be done"
)
parser.add_argument(
    "framework",
    help=f"framework to build ({FRAMEWORKS + ['all']})",
)
args = parser.parse_args()

if args.framework == "all":
    frameworks = FRAMEWORKS
elif args.framework in FRAMEWORKS:
    frameworks = [args.framework]
else:
    raise ValueError(f"unknown framework {args.framework}")

for framework in frameworks:
    dockerfile = script_dir / f"Dockerfile-{framework}"
    if not dockerfile.exists():
        print("make sure you run this script from the root of the armory repo")
        raise ValueError(f"Dockerfile not found: {dockerfile}")

    cmd = [
        "docker",
        "build",
        "--file",
        str(dockerfile),
        "--tag",
        f"twosixarmory/{framework}:{armory.__version__}",
        "--build-arg",
        f"base_image_tag={args.base_tag}",
        "--build-arg",
        f"armory_version={armory.__version__}",
        "--force-rm",
    ]
    if args.no_cache:
        cmd.append("--no-cache")
    if not args.no_pull:
        cmd.append("--pull")

    cmd.append(os.getcwd())

    print("about to run: ", " ".join(cmd))
    if args.dry_run:
        print("dry-run requested, not executing build")
    else:
        subprocess.run(cmd)
