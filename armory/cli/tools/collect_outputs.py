import argparse
import os

from armory import paths
from armory.cli.tools.utils import _debug
from armory.logs import log, update_filters


def collect_armory_outputs(command_args, prog, description):
    """TODO: Fill"""
    parser = argparse.ArgumentParser(
        prog=prog,
        description=description,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--glob",
        "-g",
        type=str,
        help="Glob pattern to match outputs. Defaults to `*`.",
        default="*",
    )
    _debug(parser)
    args = parser.parse_args(command_args)
    update_filters(args.log_level, args.debug)

    # We need to check both output dirs since `--no-docker` isn't passed
    _host = os.path.isdir(paths.HostPaths().output_dir)
    _dock = os.path.isdir(paths.DockerPaths().output_dir)
    if not _host and not _dock:
        raise ValueError("No output dir found. Please run a task first.")
    outputs = os.listdir(paths.HostPaths().output_dir) if _host else []
    outputs += os.listdir(paths.DockerPaths().output_dir) if _dock else []
    if len(outputs) == 0:
        raise ValueError("No outputs found. Please run a task first.")
    log.debug(f"Found {len(outputs)} outputs:\n{outputs}")

    breakpoint()
