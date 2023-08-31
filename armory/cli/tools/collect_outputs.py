import argparse
import json
import os
from pathlib import Path
from typing import Generator

from armory import paths
from armory.cli.tools.utils import _debug
from armory.logs import log, update_filters


def _clean(output_dir):
    breakpoint()
    clean_dir = output_dir / ".cleaned"
    # get all directories containing _only_ armory-log.txt and colored-log.txt
    empty_dirs = [
        d
        for d in output_dir.rglob("**/")
        if d.name != "saved_samples"
        and not d.is_relative_to(clean_dir)
        and len(list(d.glob("*"))) == 2
        and any([f.name == "armory-log.txt" for f in d.glob("*")])
        and any([f.name == "colored-log.txt" for f in d.glob("*")])
    ]
    log.info(
        f"Found {len(empty_dirs)} empty directories to clean. Moving to {clean_dir}"
    )
    for d in empty_dirs:
        target = clean_dir / d.relative_to(output_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        d.rename(target)


def _get_run_name(d) -> str:
    filepath = d["config"]["sysconfig"].get("filepath", None)
    if filepath is None:
        filepath = d["config"]["sysconfig"].get("config_filepath", None)
    return f"[%s]({filepath})"


def _get_dataset_name(d) -> str:
    name = d["config"]["dataset"].get("name", None)
    if name is None:
        name = d["config"]["dataset"].get("test", {}).get("name", None)
    return name


def _get_attack_kwargs(d) -> str:
    name_map = {
        "learning_rate": "lr",
        "learning_rate_depth": "lrd",
        # "max_iter": "max_iter",
        "patch_base_image": "base_image",
    }
    ignore_keys = {"optimizer", "targeted", "verbose", "patch_base_image"}
    _ = d["config"]["attack"]["kwargs"].pop("patch_mask", None)
    return "\n".join(
        [
            f"{name_map.get(k, k)}={v}"
            for k, v in d["config"]["attack"]["kwargs"].items()
            if k not in ignore_keys
        ]
    )


def _get_mAP(d) -> str:
    builder = list()
    if "benign_carla_od_AP_per_class" in d["results"]:
        benign_AP = d["results"]["benign_carla_od_AP_per_class"][0]["mean"]
        benign_AP = f"{round(benign_AP * 100) / 100:.2f}"
        builder.append(benign_AP)
    if "adversarial_carla_od_AP_per_class" in d["results"]:
        adv_AP = d["results"]["adversarial_carla_od_AP_per_class"][0]["mean"]
        adv_AP = f"{round(adv_AP * 100) / 100:.2f}"
        builder.append(adv_AP)
    return "/".join(builder)


CARLA_HEADERS = {
    "Run": _get_run_name,
    "Defense": lambda d: "",
    "Dataset": _get_dataset_name,
    "Attack": lambda d: d["config"]["attack"]["name"],
    "Attack Params": _get_attack_kwargs,
    "mAP": _get_mAP,
}


def _parse_carla_adversarial_patch(json_data, filepath) -> Generator[str, None, None]:
    """return a row of the table for a CARLA adversarial patch attack"""
    row = [None] * len(CARLA_HEADERS)
    for i, lamb in enumerate(CARLA_HEADERS.values()):
        try:
            value = lamb(json_data)
        except Exception as e:
            log.error(f"Error parsing {filepath}:\n{e}")
            breakpoint()

        if "%s" in value:
            value = value % Path(filepath).name
        yield value
        row[i] = value
    log.debug(f"Parsed {filepath}:\n{row}")


HEADERS = {
    _parse_carla_adversarial_patch: CARLA_HEADERS,
}


PARSERS = {
    "default": _parse_carla_adversarial_patch,
    "CARLAAdversarialPatchPyTorch": _parse_carla_adversarial_patch,
}


def _parse_markdown_table(headers, rows):
    """Return a markdown table string"""
    # get column widths
    widths = [
        min(max(len(h), max([len(r[i]) for r in rows])), 30)
        for i, h in enumerate(headers)
    ]
    # create header
    header = "|".join([f"{h:<{widths[i]}}" for i, h in enumerate(headers)]) + "|"
    # create separator
    separator = "|".join(["-" * w for w in widths]) + "|"
    # create rows
    rows = "\n".join(
        ["|".join([f"{r[i]:<{widths[i]}}" for i in range(len(r))]) + "|" for r in rows]
    )
    return "\n".join([header, separator, rows])


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
        help="Clean up all failed runs (directories containing _only_ {armory,colored}-log.txt). "
        + "Moves them to a new directory called .cleaned.",
    )
    parser.add_argument(
        "--absolute",
        action="store_true",
        help="Use absolute path for hyperlinks in the output tables.",
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
    output_dir = Path(
        paths.HostPaths().output_dir if _host else paths.DockerPaths().output_dir
    )
    if args.clean:
        _clean(output_dir)

    # get json results files
    results = list(
        filter(
            lambda p: p.parent.name != "saved_samples",
            output_dir.rglob("[!.]" + args.glob),
        )
    )

    # parse them into tables
    tables = {
        _parse_carla_adversarial_patch: [],
    }
    for result in results:
        # load json
        with open(result, "r") as f:
            json_data = json.load(f)
        # parse json
        parser = PARSERS.get(json_data["config"]["attack"]["name"], PARSERS["default"])
        row = list(parser(json_data, result))
        tables[parser].append(row)

    for parse_fn, table_rows in tables.items():
        headers = HEADERS[parse_fn]
        table = _parse_markdown_table(headers, table_rows)
        log.info(f"Results for {parse_fn.__name__}:\n{table}")
        breakpoint()
