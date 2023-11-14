import argparse
from collections import defaultdict
from functools import lru_cache, reduce
import json
import operator
import os
from pathlib import Path
import re
import time
from typing import Generator, Optional, Tuple, Union

from PIL import Image

from armory import paths
from armory.cli.tools.utils import _debug, simple_progress_bar, human_sort
from armory.logs import log, update_filters


class NeedMoreInfoError(Exception):
    pass


def _clean(output_dir: str, expiration: int = 6 * 60 * 60):
    output_dir = Path(output_dir)
    clean_dir = output_dir / ".cleaned"
    # get all directories containing _only_ armory-log.txt and colored-log.txt
    # ignore directories modified within the last 6 hours (to prevent active runs from being deleted)
    empty_dirs = [
        d
        for d in output_dir.rglob("**/")
        if d.name != "saved_samples"
        and not d.is_relative_to(clean_dir)
        and len(list(d.glob("*"))) == 2
        and any([f.name == "armory-log.txt" for f in d.glob("*")])
        and any([f.name == "colored-log.txt" for f in d.glob("*")])
        and d.stat().st_mtime - time.time() > expiration
    ]
    log.info(
        f"Found {len(empty_dirs)} empty directories to clean. Moving to {clean_dir}"
    )
    for d in empty_dirs:
        target = clean_dir / d.relative_to(output_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        d.rename(target)
    # delete all pairs of .png files NAME.png and NAME_adv.png where both files are equivalent
    # (i.e. NAME.png is a copy of NAME_adv.png)
    log.info("Cleaning up duplicate poisoned .png files")

    pngs = list(
        filter(
            lambda p: len(list(p.parent.parent.glob("*Poison*"))) > 0,
            output_dir.rglob("*x.png"),
        )
    )
    log.debug(f"Found {len(pngs)} .png files to check for duplicates.")
    unlinks = set()
    for png in simple_progress_bar(pngs, msg="Checking equality..."):
        adv = png.parent / (png.stem + "_adv.png")
        if adv.exists() and Image.open(png).tobytes() == Image.open(adv).tobytes():
            unlinks.add(png)
            unlinks.add(adv)
    log.info(
        f"Found {len(pngs) - len(unlinks) // 2}/{len(pngs)} non-duplicate .png pairs. Moving the rest to {clean_dir}/pngs/"
    )
    for png in simple_progress_bar(unlinks, msg="Moving files..."):
        target = clean_dir / png.relative_to(output_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        png.rename(target)
    log.info(
        f"Moved {len(unlinks) // 2} pairs of duplicate .png files to {clean_dir}/pngs/"
    )


def _get_run_name(d) -> str:
    fallthrough = [
        "config.sysconfig.filepath",
        "config.sysconfig.config_filepath",
        "config.sysconfig.output_filename",
    ]
    filepath = None
    while filepath is None and len(fallthrough) > 0:
        filepath = _get_dict_value(fallthrough.pop(0), d, default_value=None)
    if filepath is None or filepath == "-":
        # can't find config filepath, just return result filepath as %s
        return "%s"
    return f"[{filepath}]({filepath})"


def _get_dataset_name(d) -> str:
    name = d["config"]["dataset"].get("name", None)
    if name is None:
        name = d["config"]["dataset"].get("test", {}).get("name", None)
    return name


_IGNORE_KWARGS = {
    "optimizer",
    "targeted",
    "verbose",
    "patch_base_image",
    "patch",
    "device_name",
    "batch_size",
}


def _get_attack_kwargs(d) -> str:
    name_map = {
        "learning_rate": "lr",
        "learning_rate_depth": "lrd",
        # "max_iter": "max_iter",
        "patch_base_image": "base_image",
        "learning_rate_schedule": "lrs",
    }

    patch_mask = _get_dict_value(
        "config.attack.kwargs.patch_mask", d, default_value=None
    )
    # make the url as short as possible
    path_replacements = ["https://", "http://", "?raw=true", "www."]
    patch_mask_str = ""
    if isinstance(patch_mask, dict):
        shape = patch_mask.get("shape", None)
        path = patch_mask.get("path", None)
        if shape is not None:
            patch_mask_str += f"shape={shape}"
        elif path is not None:
            for r in path_replacements:
                path = path.replace(r, "")
            patch_mask_str += f"path={path}"
        invert = patch_mask.get("invert", None)
        if invert is not None:
            patch_mask_str += f",invert={invert}"
        fill = patch_mask.get("fill", None)
        if fill is not None:
            for r in path_replacements:
                fill = fill.replace(r, "")
            patch_mask_str += f",fill={fill}"
        log.debug(f"patch_mask: {patch_mask_str}")
        d["config"]["attack"]["kwargs"]["patch_mask"] = patch_mask_str

    return " ".join(
        [
            f"{name_map.get(k, k)}={v}"
            for k, v in d["config"]["attack"]["kwargs"].items()
            if k not in _IGNORE_KWARGS
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


def _getitem(d, getters, **kwargs):
    try:
        return reduce(operator.getitem, getters, d)
    except (TypeError, KeyError) as e:
        if "default_value" in kwargs:
            return kwargs["default_value"]
        raise e


def _get_dict_value(string: str, d: dict = None, **kwargs):
    """Return a lambda that indexes into a nested dict using the supplied string."""
    # this function exists so I don't have to type out brackets and quotes
    if d is None:
        getters = string.split(".")
        getters = [int(s) if s.isdigit() else s for s in getters]
        return lambda d: _getitem(d, getters, **kwargs)
    try:
        return _get_dict_value(string, **kwargs)(d)
    except KeyError as e:
        if "default_value" in kwargs:
            return kwargs["default_value"]
        raise e


def _round_rate(rate: Optional[Union[float, list]], places: int = 2) -> float:
    if rate is None:
        return None
    if isinstance(rate, list):
        return _round_rate(sum(rate) / len(rate), places)
    return round(rate, places)


def _unknown_mean_key_helper(d, key: str, places=2) -> str:
    # Don't know which key it will be under, so try both
    if "_mean" not in key:
        raise ValueError(f"Key {key} does not contain '_mean'")
    try:
        value = _get_dict_value(key, d)[0]
    except KeyError:
        try:
            value = _get_dict_value(key.replace("_mean", ""), d)[0]
        except KeyError:
            value = None
    return _round_rate(value, places=places)


def _benign_adversarial_helper(d, key: str, places=2) -> str:
    benign_rate = _unknown_mean_key_helper(
        d, f"results.benign_mean_{key}", places=places
    )
    adversarial_rate = _unknown_mean_key_helper(
        d, f"results.adversarial_mean_{key}", places=places
    )
    return f"{benign_rate}/{adversarial_rate}"


def _tide_helper(d, key: str, places=4) -> str:
    benign_rate, adversarial_rate = (
        _round_rate(
            _get_dict_value(
                f"results.{which}_object_detection_mAP_tide.0.errors.{key}",
                d,
                default_value=None,
            ),
            places=places,
        )
        for which in ("benign", "adversarial")
    )
    return f"{benign_rate}/{adversarial_rate}"


def _benign_poisoned_helper(d, key: str, places=4) -> str:
    benign_rate, poisoned_rate = (
        _round_rate(
            _get_dict_value(
                f"results.accuracy_on_{which}_{key}.0",
                d,
                default_value=None,
            ),
            places=places,
        )
        for which in ("benign", "poisoned")
    )
    return f"{benign_rate}/{poisoned_rate}"


def _get_seconds_from_runtime(runtime: str) -> int:
    """Return the number of seconds from a runtime string in HhMmSs format.
    Note format can also be in HhMm or MmSs format."""
    # use regex to parse runtime
    pattern = r"((?P<hours>\d+)h)?((?P<minutes>\d+)m)?((?P<seconds>\d+)s)?"
    match = re.match(pattern, runtime)
    if match is None:
        raise ValueError(f"Could not parse runtime {runtime}")
    hours = match.group("hours")
    minutes = match.group("minutes")
    seconds = match.group("seconds")
    if hours is None:
        hours = 0
    if minutes is None:
        minutes = 0
    if seconds is None:
        seconds = 0
    return int(hours) * 60 * 60 + int(minutes) * 60 + int(seconds)


def _get_runtime(d: dict, filepath: Optional[str] = None) -> str:
    """Return the runtime of the run in HhMmSs format."""
    if filepath is None:
        filepath = _split_markdown_href(_get_run_name(d))[1]
    if filepath is None or not (filepath := Path(filepath)).exists():
        raise NeedMoreInfoError("Could not find filepath")
    if filepath.suffix == ".json":
        logfile = filepath.parent / "armory-log.txt"
    elif filepath.name == "armory-log.txt":
        logfile = filepath
    if not logfile.exists():
        raise NeedMoreInfoError(f"Could not find armory-log.txt from {filepath}")
    # tail -n 1 $file | cut -d" " -f3
    with open(logfile, "r") as f:
        last_line = f.readlines()[-1]
    return last_line.split(" ")[2]


def _parse_json_data(
    headers, json_data, filepath, **kwargs
) -> Generator[str, None, None]:
    """return a row of the table given a dict of headers (& parsers) and json_data"""
    row = [None] * len(headers)
    for i, parse in enumerate(headers.values()):
        try:
            value = parse(json_data)
        except NeedMoreInfoError:
            # try again with filepath
            value = parse(json_data, filepath=filepath)
        except Exception as e:
            log.error(f"Error parsing {list(headers.keys())[i]} from {filepath}:\n{e}")
            log.error(f"json_data:\n{json_data}")
            raise e

        if value is not None and isinstance(value, str) and "%s" in value:
            value %= filepath
        yield value
        row[i] = value
    log.debug(f"Parsed {filepath}:\n{row}")


CARLA_HEADERS = {
    "Run": _get_run_name,
    "Defense": _get_dict_value("config.defense.name", default_value=None),
    "Dataset": _get_dataset_name,
    "Attack": _get_dict_value("config.attack.name", default_value=None),
    "Attack Params": _get_attack_kwargs,
    "mAP": _get_mAP,
    "Disappearance Rate": lambda d: _benign_adversarial_helper(
        d, "carla_od_disappearance_rate"
    ),
    "Hallucinations per Img": lambda d: _benign_adversarial_helper(
        d, "carla_od_hallucinations_per_image", places=1
    ),
    "Misclassification Rate": lambda d: _benign_adversarial_helper(
        d, "carla_od_misclassification_rate", places=3
    ),
    "True Positive Rate": lambda d: _benign_adversarial_helper(
        d, "carla_od_true_positive_rate"
    ),
    "dAP_Bkg": lambda d: _tide_helper(d, "main.dAP.Bkg"),
    "dAP_Both": lambda d: _tide_helper(d, "main.dAP.Both"),
    "dAP_Cls": lambda d: _tide_helper(d, "main.dAP.Cls"),
    "dAP_Dupe": lambda d: _tide_helper(d, "main.dAP.Dupe"),
    "dAP_Loc": lambda d: _tide_helper(d, "main.dAP.Loc"),
    "dAP_Miss": lambda d: _tide_helper(d, "main.dAP.Miss"),
    "dAP_FalseNeg": lambda d: _tide_helper(d, "special.dAP.FalseNeg"),
    "dAP_FalsePos": lambda d: _tide_helper(d, "special.dAP.FalsePos"),
    "runtime": _get_runtime,
    "output": lambda d: f"[result json]({_get_dict_value('config.attack.name', default_value='results', d=d)}/%s)",
}


SLEEPER_AGENT_HEADERS = {
    "Run": _get_run_name,
    "Defense": _get_dict_value("config.defense.name", default_value=None),
    "Dataset": _get_dataset_name,
    "Attack": _get_dict_value("config.attack.name", default_value=None),
    "Attack Params": _get_attack_kwargs,
    "Poison %": _get_dict_value("config.adhoc.fraction_poisoned", default_value=None),
    "Attack Success Rate": _get_dict_value(
        "results.attack_success_rate.0", default_value=None
    ),
    "Accuracy (Benign/Poisoned)": lambda d: _benign_poisoned_helper(
        d, "test_data_all_classes"
    ),
    "runtime": _get_runtime,
    "output": lambda d: "[result json](%s)",
}


CIFAR_HEADERS = {
    "Run": _get_run_name,
    "Defense": _get_dict_value("config.defense.name", default_value=None),
    "Dataset": _get_dataset_name,
    "Attack": _get_dict_value("config.attack.name", default_value=None),
    "Attack Params": _get_attack_kwargs,
    "Mean Categorical Accuracy": lambda d: _benign_adversarial_helper(
        d, "categorical_accuracy"
    ),
    "runtime": _get_runtime,
    # "output": lambda d: f"[result json]({_get_dict_value('config.attack.name', default_value='results', d=d)}/%s)",
    "output": lambda d: "[result json](%s)",
}


HEADERS = {
    "CARLAAdversarialPatchPyTorch": CARLA_HEADERS,
    "SleeperAgentAttack": SLEEPER_AGENT_HEADERS,
    "MITMPoisonSleeperAgent": SLEEPER_AGENT_HEADERS,
    "SleeperAgentScenario": SLEEPER_AGENT_HEADERS,
    "ProjectedGradientDescent": CIFAR_HEADERS,
    "PGDPatch": CIFAR_HEADERS,
}


def _parse_markdown_table(headers, rows):
    """Generate a markdown table from headers and rows.
    Optionally supply list of lists of headers to return tables
    stacked vertically separated by a newline."""
    if isinstance(headers[0], list):
        return "\n\n".join([_parse_markdown_table(h, r) for h, r in zip(headers, rows)])
    rows = [[str(r) for r in row] for row in rows]
    column_widths = [max(map(len, col)) for col in zip(*rows, headers)]
    headers = [f"{h:<{w}}" for h, w in zip(headers, column_widths)]
    rows = [[f"{r:<{w}}" for r, w in zip(row, column_widths)] for row in rows]
    table = "\n".join(
        [
            "|".join(headers),
            "|".join(["-" * w for w in column_widths]),
            *["|".join(map(str, row)) for row in rows],
        ]
    )
    return table


@lru_cache(maxsize=None)
def _get_output_dir() -> str:
    """Return the output directory"""
    _host = os.path.isdir(paths.HostPaths().output_dir)
    _dock = os.path.isdir(paths.DockerPaths().output_dir)
    if not _host and not _dock:
        raise ValueError("No output dir found. Please run a task first.")
    if _host and _dock:
        raise ValueError(
            "Found both host and docker output dirs, cannot determine which to use."
        )
    return paths.HostPaths().output_dir if _host else paths.DockerPaths().output_dir


def _add_parser_args(parser, output_dir: str = _get_output_dir()):
    parser.add_argument(
        "--glob",
        "-g",
        type=str,
        help="Glob pattern to match json outputs. Defaults to `*.json`.",
        default="*.json",
    )
    output_default = Path(output_dir).parent / "results/%s.md"
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=output_default,
        help=f"Path to output tables. Defaults to {output_default.parent / 'ATTACK.md'} where str format placeholder is replaced with ATTACK name if supplied.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean up all failed runs (directories containing _only_ {armory,colored}-log.txt).\n"
        + "Moves them to a new directory called .cleaned.",
    )
    parser.add_argument(
        "--unify",
        type=str,
        nargs="*",
        help="Unify results from multiple attacks into a single markdown file. "
        + "Takes a list of attack names to unify.\n"
        + "Defaults to all attacks if no attack is supplied. "
        + "Does not output individual tables. ",
        metavar="ATTACK",
    )
    parser.add_argument(
        "--collate",
        "-c",
        type=str,
        nargs="?",
        default=False,
        help="Combine attack results based on the supplied kwarg. Defaults to `config.metric.task`.",
        metavar="KWARG",
    )
    parser.add_argument(
        "--absolute",
        action="store_true",
        help="Use absolute path for hyperlinks in the output tables. Does not create symbolic links.",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path(".").absolute(),
        help="Path to link run configs from. Defaults to current directory.",
    )
    parser.add_argument(
        "--default",
        type=str,
        default="CARLAAdversarialPatchPyTorch",
        help="Default attack to use for headers. Defaults to CARLAAdversarialPatchPyTorch.",
    )
    parser.add_argument(
        "--sort",
        type=str,
        nargs="?",
        default=False,
        help="Sort results by the supplied header(s).",
        metavar="HEADER",
    )
    parser.add_argument(
        "--filter",
        type=str,
        help="Filter results to only those matching the supplied regex.",
    )
    parser.add_argument(
        "--ignore-header",
        "-x",
        type=str,
        nargs="+",
        help="Ignore the supplied headers when parsing results.",
    )
    kwargs_group = parser.add_mutually_exclusive_group()
    kwargs_group.add_argument(
        "--ignore-kwargs",
        type=str,
        nargs="+",
        help="Ignore the supplied kwargs when parsing attack params.",
    )
    kwargs_group.add_argument(
        "--keep-kwargs",
        type=str,
        nargs="+",
        help="Only keep the supplied kwargs when parsing attack params.",
    )
    parser.add_argument(
        "--ignore-short-runs",
        type=int,
        nargs="?",
        default=False,
        help="Ignore runs with runtime less than this value (in minutes). Defaults to 1 minute.",
    )
    _debug(parser)


def _split_markdown_href(name: str) -> Tuple[Optional[str], Optional[str]]:
    """Split a name into a name and a path.
    Run name is usually result filepath and path is config filepath"""
    if name is None:
        return None, None
    pattern = r"\[([^]]+)\]\(([^)]+)\)"
    hits = re.findall(pattern, name)
    if len(hits) == 0:
        return name, None
    if len(hits) > 1:
        raise ValueError(f"Found multiple links in name {name}")
    name, path = hits[0]
    return name, path


def _safe_symlink(src: Path, dst: Path):
    if isinstance(src, str):
        src = Path(src)
    if isinstance(dst, str):
        dst = Path(dst)
    if dst.is_symlink():
        if src.resolve() == dst.resolve():
            log.debug(f"Symlink {dst} already exists.")
            return
        else:
            log.warning(
                f"Symlink {dst} used to point to {dst.resolve()}, but now points to {src.resolve()}."
            )
            os.remove(dst)
    os.symlink(src, dst)
    log.debug(f"Created symlink {dst} -> {src}")


def _create_symlinks_and_update_rows(
    headers, rows, output_path, config_dir, ignore_short_runs, absolute, **kwargs
):
    # check runtime
    remove_indices = []
    if "runtime" in headers:
        for row in rows:
            runtime = row[headers.index("runtime")]
            if (
                runtime is not None
                and _get_seconds_from_runtime(runtime) < ignore_short_runs
            ):
                remove_indices.append(rows.index(row))
    for i in sorted(remove_indices, reverse=True):
        del rows[i]

    if "output" in headers:
        log.debug(f"Creating symlinks to results in {output_path.with_suffix('')}/")
    if absolute:
        log.debug("Using absolute paths for hyperlinks.")
    for row in rows:
        if "Run" in headers:
            conf = row[headers.index("Run")]
            conf_name, conf_path = _split_markdown_href(conf)
            if conf_path is not None and not absolute:
                conf_path = Path(conf_path)
                if not conf_path.exists():
                    conf_path = config_dir.absolute() / conf_path
                    if not conf_path.exists():
                        log.warning(
                            f"Config path {conf_path} does not exist, skipping symlink."
                        )
                        continue
                conf_path = os.path.relpath(conf_path, config_dir)
                if Path(conf_name).is_file():
                    conf_name = conf_path
                row[0] = f"[{conf_name}]({conf_path})"

        if "output" not in headers:
            continue
        res = row[headers.index("output")]
        if res is None:
            continue
        result_name, result_path = _split_markdown_href(res)
        # if we have a result_path, symlink it
        if result_path is not None:
            result_path = Path(result_path)
            if not result_path.exists():
                result_path = Path(_get_output_dir()) / result_path
                if not result_path.exists():
                    log.warning(
                        f"Result path {result_path} does not exist, skipping symlink."
                    )
                    continue
            os.makedirs(output_path.with_suffix(""), exist_ok=True)
            symlink_path = output_path.with_suffix("") / result_path.parent.name
            _safe_symlink(
                result_path.parent,
                symlink_path,
            )
            # change result path to symlink
            if not absolute:
                symlink_json_path = symlink_path / result_path.name
                result_path = symlink_json_path.relative_to(
                    os.path.commonpath([output_path, symlink_json_path])
                )
                row[headers.index("output")] = f"[{result_name}]({result_path})"


def _write_results(attacks, headers, rows, output, absolute, config_dir, **kwargs):
    ignore_short_runs = kwargs.pop("ignore_short_runs", 1)
    attack_name = "combined" if len(attacks) > 1 else attacks[0]
    output_path = Path(str(output) % attack_name if "%s" in str(output) else output)
    os.makedirs(output_path.parent, exist_ok=True)
    if not absolute:
        # _create_symlinks_and_update_rows(
        #     headers, rows, output_path, config_dir, ignore_short_runs
        # )
        _create_symlinks_and_update_rows(**locals())
    table = _parse_markdown_table(headers, rows)
    # log.debug(f"Results for {attacks}:\n{table}")
    with open(output_path, "w") as f:
        f.write(table)
    lines = len(table.split("\n")) - 1
    log.info(f"Wrote {lines} {attack_name} results to {output_path}.")


def collect_armory_outputs(command_args, prog, description):
    """Collect results from armory output_directory and organize into tables."""
    # We need to check both output dirs since `--no-docker` isn't passed
    output_dir = _get_output_dir()
    parser = argparse.ArgumentParser(
        prog=prog,
        description=description,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    _add_parser_args(parser, output_dir=output_dir)
    args = parser.parse_args(command_args)
    update_filters(args.log_level, args.debug)
    if args.collate is not False and args.collate is None:
        args.collate = "config.scenario.name"

    for at in args.unify or []:
        if at not in HEADERS:
            raise ValueError(f"Unrecognized attack {at}")

    if args.default not in HEADERS:
        raise ValueError(f"Unrecognized default attack {args.default}")

    if args.clean:
        return _clean(output_dir)

    if args.ignore_kwargs is not None:
        _IGNORE_KWARGS.update(args.ignore_kwargs)

    if args.keep_kwargs is not None:
        _IGNORE_KWARGS.difference_update(args.keep_kwargs)

    if args.ignore_header is not None:
        for _headers in HEADERS.values():
            for h in args.ignore_header:
                _headers.pop(h, None)

    if args.ignore_short_runs is not False and args.ignore_short_runs is None:
        args.ignore_short_runs = 1
    if args.ignore_short_runs:
        args.ignore_short_runs *= 60

    # get json results files
    log.info(f"Recursive globbing for {args.glob} in {output_dir}")
    results = list(
        filter(
            lambda p: p.parent.name != "saved_samples" and p.resolve() == p,
            Path(output_dir).rglob("*/" + args.glob),
        )
    )
    log.info(f"Found {len(results)} results files.")

    # parse them into tables
    tables = defaultdict(list)
    collation = defaultdict(set)
    for result in results:
        log.debug(f"Loading {result}")
        # load json
        with open(result, "r") as f:
            json_data = json.load(f)
        # parse json
        attack_name = json_data["config"]["attack"]["name"]
        row = list(
            _parse_json_data(
                HEADERS.get(attack_name, HEADERS[args.default]),
                json_data,
                result,
            )
        )
        if args.filter is not None:
            if not re.search(args.filter, row[0]):
                log.debug(f"Skipping {result} due to filter")
                continue
        if args.collate is not False:
            try:
                tgt_collate = reduce(
                    operator.getitem, args.collate.split("."), json_data
                )
            except (KeyError, TypeError):
                log.debug(f"Could not collate {result} on {args.collate}")
            else:
                collation[attack_name].add(str(tgt_collate))
        tables[attack_name].append(row)
    log.info(f"Parsed {len(tables)} results.")

    # collate results
    if args.collate is not False:
        for tgt_collate, attacks in collation.items():
            # make sure all attacks have the same headers
            first = next(iter(attacks))
            if any(
                HEADERS.get(first, HEADERS[args.default])
                is not HEADERS.get(a, HEADERS[args.default])
                for a in attacks
            ):
                raise ValueError(
                    f"Collation failed: attacks {attacks} have different headers."
                )

            for attack in attacks:
                log.info(f"Collating {attack} into {tgt_collate}")
                tables[tgt_collate].extend(tables[attack])
                del tables[attack]

    # sort results
    if args.sort is not False:
        if args.sort is None:
            args.sort = ["Run"]
        log.debug(f"Sorting results by {args.sort}")
        for attack, table_rows in tables.items():
            headers = list(HEADERS.get(attack, HEADERS[args.default]).keys())
            sort_indices = [headers.index(s) for s in args.sort]
            tables[attack] = human_sort(
                table_rows, key=lambda x: "".join(tuple(x[i] for i in sort_indices))
            )

    if args.unify is not None:
        if len(args.unify) == 0:
            attacks = tables.keys()
        else:
            attacks = args.unify
        headers = [list(HEADERS.get(a, HEADERS[args.default]).keys()) for a in attacks]
        rows = [tables[a] for a in attacks]
        _write_results(attacks, headers, rows, **vars(args))
    else:
        for attack, rows in sorted(
            tables.items(), key=lambda x: len(x[1]), reverse=True
        ):
            headers = list(HEADERS.get(attack, HEADERS[args.default]).keys())
            _write_results([attack], headers, rows, **vars(args))
