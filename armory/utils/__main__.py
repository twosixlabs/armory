import armory.__main__ as armory_main
import sys
import armory
import argparse
from armory.__main__ import _debug, Command
from typing import Union, Tuple
import os
from pathlib import Path

try:
    import numpy as np
except ImportError:
    raise ImportError(
        "numpy is required to convert depth images.\n"
        "Please install with `pip install numpy`."
    )
try:
    import matplotlib.pyplot as plt
    from matplotlib import widgets
except ImportError:
    raise ImportError(
        "matplotlib is required to show depth images.\n"
        "Please install with `pip install matplotlib`."
    )
try:
    import PIL.Image
    from PIL import Image
except ImportError:
    raise ImportError(
        "Pillow is required to convert depth images.\n"
        "Please install with `pip install pillow`."
    )
try:
    from armory.art_experimental.attacks.carla_obj_det_utils import rgb_depth_to_linear
    from armory.art_experimental.attacks.carla_obj_det_utils import linear_to_log
except ImportError:
    print(
        "WARNING: armory.art_experimental.attacks.carla_obj_det_utils cannot be imported. Using stubbed functions."
    )

    def rgb_depth_to_linear(r, g, b):
        """
        Converts rgb depth values between [0,1] to linear depth in meters.
        r, g, b: three scalars or arrays with the same shape in [0,1]
        returns: scalar or array, nonnegative
        """
        r_ = r * 255.0
        g_ = g * 255.0
        b_ = b * 255.0
        depth_m = r_ + g_ * 256 + b_ * 256 * 256
        depth_m = depth_m * 1000.0 / (256**3 - 1)
        return depth_m

    # Reference: https://github.com/carla-simulator/data-collector/blob/master/carla/image_converter.py
    def linear_to_log(depth_meters):
        """
        Convert linear depth in meters to logorithmic depth between [0,1]
        depth_meters: scalar or array, nonnegative
        returns: scalar or array in [0,1]
        """
        # Reference https://carla.readthedocs.io/en/latest/ref_sensors/#depth-camera
        normalized_depth = depth_meters / 1000.0
        # Convert to logarithmic depth.
        depth_log = 1.0 + np.log(normalized_depth) / 5.70378
        depth_log = np.clip(depth_log, 0.0, 1.0)
        return depth_log


def load_image(path: Union[str, Path]) -> PIL.Image:
    if isinstance(path, str):
        path = Path(path)
    if not path.is_file():
        raise ValueError(f"{path} does not exist")
    return PIL.Image.open(path)


def load_images(path: Union[str, Path, list]) -> list:
    if isinstance(path, list):
        return [load_image(p) for p in path]
    return [load_image(path)]


def convert_image(image: PIL.Image):  # -> Tuple[PIL.Image, PIL.Image]:
    r, g, b = [np.array(x) for x in image.split()]
    depth_m = rgb_depth_to_linear(r / 255.0, g / 255.0, b / 255.0)
    img_linear = PIL.Image.fromarray(depth_m)
    img_log = PIL.Image.fromarray(linear_to_log(depth_m) * 255)
    return img_linear, img_log


def rgb_depth_convert(command_args, prog, description):
    parser = argparse.ArgumentParser(
        prog=prog,
        description=description,
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "-i", "--input", type=Path, nargs="+", help="Path to depth image to convert"
    )
    # parser.add_argument(
    #     "-o", "--output", type=str, help="Path to save converted depth image"
    # )
    parser.add_argument(
        "-s",
        "--show",
        action="store_true",
        default=True,
        help="Show converted depth image using matplotlib"
        + "\n  Note:\trequires X11 forwarding & GUI backend"
        + "\n\tinstalled (ex. PyQt5) to work over ssh."
        + "\n  See:\thttps://matplotlib.org/faq/usage_faq.html#what-is-a-backend",
    )
    # parser.add_argument(
    #     "--format",
    #     type=str,
    #     default="original",
    #     choices=["log", "linear", "original"],
    # )

    _debug(parser)
    args = parser.parse_args(command_args)
    armory.logs.update_filters(args.log_level, args.debug)

    if args.input is None:
        parser.error("input path is required")

    if not args.show:
        parser.error("outputting to a file is not yet supported, must use --show")

    images = load_images(args.input)
    print(f"Loaded {len(images)} images from {args.input}")
    # Create a figure and plot the initial image
    fig, ax = plt.subplots()
    im = ax.imshow(images[0])
    if len(images) > 1:
        # Add slider to switch image
        ax_slider = fig.add_axes([0.1, 0, 0.2, 0.05])
        slider = widgets.Slider(
            ax_slider, "Image", 0, len(images) - 1, valinit=0, valstep=1
        )

        def update_image(event):
            im.set_array(images[int(slider.val)])
            fig.canvas.draw_idle()

        slider.on_changed(update_image)

    # if args.format == "original":
    #     im = ax.imshow(image)
    # elif args.format == "log":
    #     im = ax.imshow(img_log)
    # elif args.format == "linear":
    #     im = ax.imshow(img_linear)
    # else:
    #     raise ValueError(f"Invalid format {args.format}")

    def show_linear(event):
        im.set_array(convert_image(images[int(slider.val)])[0])
        fig.canvas.draw_idle()

    def show_log(event):
        im.set_array(convert_image(images[int(slider.val)])[1])
        fig.canvas.draw_idle()

    def show_orig(event):
        im.set_array(images[int(slider.val)])
        fig.canvas.draw_idle()

    # Add buttons to the plot
    linear_ax = fig.add_axes([0.7, 0.9, 0.2, 0.05])
    linear_button = widgets.Button(linear_ax, "Show Linear")
    linear_button.on_clicked(show_linear)

    log_ax = fig.add_axes([0.5, 0.9, 0.2, 0.05])
    log_button = widgets.Button(log_ax, "Show Log")
    log_button.on_clicked(show_log)

    orig_ax = fig.add_axes([0.3, 0.9, 0.2, 0.05])
    orig_button = widgets.Button(orig_ax, "Show Original")
    orig_button.on_clicked(show_orig)

    # Show the plot
    plt.show()


COMMANDS = {
    "rgb-convert": (rgb_depth_convert, "converts rgb depth images to another format"),
}


def main() -> int:
    armory_main.COMMANDS = COMMANDS
    # TODO the run method now returns a status code instead of sys.exit directly
    # the rest of the COMMANDS should conform
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        print(armory_main.usage())
        sys.exit(1)
    elif sys.argv[1] in ("-v", "--version", "version"):
        print(f"{armory.__version__}")
        sys.exit(0)
    elif sys.argv[1] == "--show-docker-version-tag":
        print(to_docker_tag(armory.__version__))
        sys.exit(0)

    parser = argparse.ArgumentParser(prog="armory", usage=armory_main.usage())
    parser.add_argument(
        "command",
        metavar="<command>",
        type=str,
        help="armory command",
        action=Command,
    )
    args = parser.parse_args(sys.argv[1:2])

    func, description = COMMANDS[args.command]
    prog = f"{armory_main.PROGRAM} {args.command}"
    return func(sys.argv[2:], prog, description)


if __name__ == "__main__":
    sys.exit(main())
