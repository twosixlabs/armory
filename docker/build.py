import argparse
import subprocess
from pathlib import Path
import os
import sys

# Ensure correct location
script_dir = Path(__file__).parent
root_dir = script_dir.parent
if not (root_dir / "armory").is_dir():
    print("ERROR: make sure you run this script from the root of the armory repo")
    sys.exit(1)


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
    dist_dir   = Path(root_dir  / "dist")
    # Cleanup old builds
    if dist_dir.is_dir(): rm_tree(dist_dir)
    # Build the armory pip package
    subprocess.run(["hatch", "build", "--clean"])
    armory_wheel = [f for f in dist_dir.iterdir() if f.name.startswith("armory") and f.name.endswith(".whl")][0]
    return armory_wheel


# Parse arguments
FRAMEWORKS = ["pytorch", "pytorch-deepspeech", "tf2"]
parser = argparse.ArgumentParser(description="builds a docker image for armory")
parser.add_argument("-b", "--base-tag", help="version tag for twosixarmory", default="latest")
parser.add_argument("--no-cache", action="store_true", help="do not use docker cache")
parser.add_argument("--no-pull", action="store_true", help="do not pull latest base")
parser.add_argument("-n", "--dry-run", action="store_true", help="show what would be done")
parser.add_argument(
    "framework",
    choices=FRAMEWORKS + ["all"],
    metavar="framework",
    help=f"framework to build ({FRAMEWORKS + ['all']})",
)
args = parser.parse_args()

if args.framework == "all":
    frameworks = FRAMEWORKS
else:
    frameworks = [args.framework]

# Enable import without pip installation and retrieve armory version
sys.path.insert(0, str(root_dir))
try:
    import armory
except ModuleNotFoundError as e:
    if str(e) == "No module named 'armory'":
        print("ERROR: could not import armory. " "make sure you run this script from the root of the armory repo")
        sys.exit(1)
    raise

print("Retrieving armory version")
print(f"armory docker builder version {armory.__version__}")

# Execute docker builds
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
