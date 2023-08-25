import argparse
from pathlib import Path

from armory.cli.tools.utils import _debug
from armory.postprocessing.plot_patch_aware_carla_metric import (
    plot_mAP_by_giou_with_patch,
)


def plot_mAP_by_giou_with_patch_cli(command_args, prog, description):

    parser = argparse.ArgumentParser(
        prog=prog,
        description=description,
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Path to json. Must have 'results.adversarial_object_detection_AP_per_class_by_giou_from_patch' key.",
    )
    parser.add_argument(
        "--flavors",
        type=str,
        nargs="+",
        default=None,
        choices=["cumulative_by_max_giou", "cumulative_by_min_giou", "histogram_left"],
        help="Flavors of mAP by giou to plot. Subset of ['cumulative_by_max_giou', 'cumulative_by_min_giou', 'histogram_left'] or None to plot all.",
    )
    parser.add_argument("--headless", action="store_true", help="Don't show the plot")
    parser.add_argument(
        "--output", type=Path, default=None, help="Path to save the plot"
    )
    parser.add_argument(
        "--exclude-classes",
        action="store_true",
        help="Don't include subplot for each class.",
    )
    _debug(parser)

    args = parser.parse_args(command_args)
    plot_mAP_by_giou_with_patch(
        args.input,
        flavors=args.flavors,
        show=not args.headless,
        output_filepath=args.output,
        include_classes=not args.exclude_classes,
    )
