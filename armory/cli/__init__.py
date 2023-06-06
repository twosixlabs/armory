from armory.cli.tools.generate_shapes import generate_shapes
from armory.cli.tools.log_current_branch import log_current_branch
from armory.cli.tools.rgb_depth_convert import rgb_depth_convert

CLI_COMMANDS = {
    "get-branch": (log_current_branch, "Log the current git branch of armory"),
    "rgb-convert": (rgb_depth_convert, "Converts rgb depth images to another format"),
    "generate-shapes": (generate_shapes, "Generate shapes as png files"),
}
