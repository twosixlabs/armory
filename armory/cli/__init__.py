from armory.cli.tools.generate_shapes import generate_shapes
from armory.cli.tools.log_current_branch import log_current_branch
from armory.cli.tools.plot_mAP_by_giou_with_patch_cli import (
    plot_mAP_by_giou_with_patch_cli,
)
from armory.cli.tools.rgb_depth_convert import rgb_depth_convert

CLI_COMMANDS = {
    "get-branch": (log_current_branch, "Log the current git branch of armory"),
    "rgb-convert": (rgb_depth_convert, "Converts rgb depth images to another format"),
    "shape-gen": (generate_shapes, "Generate shapes as png files"),
    "plot-mAP-by-giou": (
        plot_mAP_by_giou_with_patch_cli,
        "Visualize the output of the metric 'object_detection_AP_per_class_by_giou_from_patch.'",
    ),
}
