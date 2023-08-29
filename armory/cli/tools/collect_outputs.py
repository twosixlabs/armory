import argparse
import os

# from pathlib import Path

from armory import paths
from armory.cli.tools.utils import _debug
from armory.logs import log, update_filters


def collect_armory_outputs(command_args, prog, description):
    """Collect results from armory output_directory and organize into tables."""
    parser = argparse.ArgumentParser(
        prog=prog,
        description=description,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--glob",
        "-g",
        type=str,
        help="Glob pattern to match json outputs. Defaults to `*.json`.",
        default="*.json",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove all failed runs (directories without a .json file).",
    )
    _debug(parser)
    args = parser.parse_args(command_args)
    update_filters(args.log_level, args.debug)

    # We need to check both output dirs since `--no-docker` isn't passed
    _host = os.path.isdir(paths.HostPaths().output_dir)
    _dock = os.path.isdir(paths.DockerPaths().output_dir)
    if not _host and not _dock:
        raise ValueError("No output dir found. Please run a task first.")
    if _host and _dock:
        raise ValueError(
            "Found both host and docker output dirs, cannot determine which to use."
        )
    # output_dir = Path(paths.HostPaths().output_dir if _host else paths.DockerPaths().output_dir)

    # # clean output directory
    # if args.clean:
    #     # get all directories without a json file in it

    # # get json files
    # leaf_dirs = [d for d in output_dir.rglob("[!.]" + args.glob) if d.is_dir()]

    outputs = os.listdir(paths.HostPaths().output_dir) if _host else []
    outputs += os.listdir(paths.DockerPaths().output_dir) if _dock else []
    if len(outputs) == 0:
        raise ValueError("No outputs found. Please run a task first.")
    log.debug(f"Found {len(outputs)} outputs:\n{outputs}")
