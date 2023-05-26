from .generate_shapes import generate_shapes
from .log_current_branch import log_current_branch
from .rgb_depth_convert import rgb_depth_convert

CLI_COMMANDS = {
    "get-branch": (log_current_branch, "log the current git branch of armory"),
    "rgb-convert": (rgb_depth_convert, "converts rgb depth images to another format"),
    "generate-shapes": (generate_shapes, "generate shapes"),
}
