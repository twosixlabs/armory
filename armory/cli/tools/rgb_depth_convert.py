import argparse
from pathlib import Path
from typing import Literal, Union

from armory.cli.tools.utils import _debug
from armory.logs import log, update_filters


def rgb_depth_convert(command_args, prog, description):
    try:
        import numpy as np
    except ImportError:
        raise ImportError(
            "numpy is required to convert depth images.\n"
            "Please install with `pip install numpy`."
        )
    try:
        from matplotlib import widgets
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required to show depth images.\n"
            "Please install with `pip install matplotlib`."
        )
    try:
        from PIL import Image
    except ImportError:
        raise ImportError(
            "Pillow is required to convert depth images.\n"
            "Please install with `pip install pillow`."
        )

    def _load_image(path: Union[str, Path]) -> Image:
        if isinstance(path, str):
            path = Path(path)
        if not path.is_file():
            raise ValueError(f"{path} does not exist")
        return Image.open(path)

    def _load_images(path: Union[str, Path, list]) -> list:
        if isinstance(path, list):
            return [_load_image(p) for p in path]
        elif isinstance(path, str):
            path = Path(path)
        if path.is_dir():
            files = path.glob("*depth.png")
            try:
                files = sorted(files, key=lambda p: int(p.stem.split("_")[1]))
            except Exception as e:
                log.error(f"Unable to sort files in {path}: {e}")
            files = [_load_image(p) for p in files]
            if len(files) > 0:
                return files
            raise ValueError(f"{path} does not contain any depth images")
        return [_load_image(path)]

    parser = argparse.ArgumentParser(
        prog=prog,
        description=description,
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "input", type=Path, nargs="+", help="Path to depth image(s) to convert"
    )
    _debug(parser)
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Don't show converted depth image using matplotlib"
        + "\n  Note:\tif using over ssh, must use --headless flag or must"
        + "\n\thave a GUI backend installed (ex. PyQt5) with X11 forwarding."
        + "\n  See:\thttps://matplotlib.org/faq/usage_faq.html#what-is-a-backend",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to save converted depth image to"
        + "\n  Note:\tCannot be used with multiple input images.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        default=False,
        help="Save converted depth image to <input_path>_<format>.png",
    )
    parser.add_argument(
        "--format",
        choices=["linear", "log"],
        default="linear",
        help="Format used with --save or --output flag.",
    )

    args = parser.parse_args(command_args)
    update_filters(args.log_level, args.debug)

    if args.input is None:
        parser.error("input path is required")

    if args.headless and (args.output is None or args.save is None):
        parser.error("Must use --output or --save flag with --headless flag")

    if args.output is not None and args.save:
        parser.error("Cannot use --output and --save flags together")

    if args.output is not None and len(args.input) > 1:
        parser.error(
            "Cannot use --output with multiple input paths, please use --save."
        )

    try:
        from armory.art_experimental.attacks.carla_obj_det_utils import (
            linear_to_log,
            rgb_depth_to_linear,
        )
    except ImportError:
        log.warning(
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

    def convert_image(
        image: Image, type: Literal["linear", "log", "original"]
    ) -> Image:
        if type == "original":
            return image
        r, g, b = [np.array(x) for x in image.split()]
        depth_m = rgb_depth_to_linear(r / 255.0, g / 255.0, b / 255.0)
        img_linear = Image.fromarray(depth_m)
        if type == "linear":
            return img_linear
        elif type == "log":
            return Image.fromarray(linear_to_log(depth_m) * 255)
        raise ValueError(f"Invalid type: {type}")

    # sort input paths
    if len(args.input) > 1:
        args.input = sorted(args.input, key=lambda p: int(p.stem.split("_")[1]))
    elif len(args.input) == 1:
        args.input = args.input[0]
    else:
        raise ValueError("No input paths found")

    images = _load_images(args.input)
    if len(images) == 0:
        raise ValueError("No images found in input paths")
    log.info(f"Loaded {len(images)} images from {args.input}")
    for i, path in enumerate(images):
        log.debug(f"{i:2d}: {Path(path.filename).name}")

    # save images if --save or --output flag is used
    if args.save or args.output is not None:
        if args.save:
            output_paths = [
                p.parent / f"{p.stem}_{args.format}.png" for p in args.input
            ]
        else:
            output_paths = [args.output]
        for i, path in enumerate(output_paths):
            log.info(f"Saving {path}")
            convert_image(images[i], args.format).convert("RGB").save(path)

    if args.headless:
        return
    # Create a figure and plot the initial image
    global display_type
    display_type = "original"
    fig, ax = plt.subplots()
    im = ax.imshow(images[0])
    if len(images) > 1:
        # Add slider to switch image
        ax_slider = fig.add_axes([0.1, 0, 0.2, 0.05])
        slider = widgets.Slider(
            ax_slider, "Image", 0, len(images) - 1, valinit=0, valstep=1
        )
        slider.valtext.set_text(f"{slider.val}: {Path(images[0].filename).name}")

        def update_image(event):
            global display_type
            im.set_array(convert_image(images[int(slider.val)], display_type))
            slider.valtext.set_text(
                f"{slider.val}: {Path(images[int(slider.val)].filename).name}"
            )
            fig.canvas.draw_idle()

        slider.on_changed(update_image)

        # Add left right buttons to adjust slider
        width, height = 0.05, 0.05
        left_ax = fig.add_axes([0.9, 0, width, height])
        left_button = widgets.Button(left_ax, "<")

        def left(event):
            slider.set_val(slider.val - 1)

        left_button.on_clicked(left)

        right_ax = fig.add_axes([0.95, 0, width, height])
        right_button = widgets.Button(right_ax, ">")

        def right(event):
            slider.set_val(slider.val + 1)

        right_button.on_clicked(right)
    else:
        slider = type("", (), {"val": 0})()

    def show(event, dtype):
        global display_type
        display_type = dtype
        im.set_array(convert_image(images[int(slider.val)], dtype))
        fig.canvas.draw_idle()

    # Add buttons to the plot
    def apply_action(but, dtype):
        but.on_clicked(lambda ev: show(ev, dtype))

    linear_ax = fig.add_axes([0.7, 0.9, 0.2, 0.05])
    linear_button = widgets.Button(linear_ax, "Show Linear")
    apply_action(linear_button, "linear")

    log_ax = fig.add_axes([0.5, 0.9, 0.2, 0.05])
    log_button = widgets.Button(log_ax, "Show Log")
    apply_action(log_button, "log")

    orig_ax = fig.add_axes([0.3, 0.9, 0.2, 0.05])
    orig_button = widgets.Button(orig_ax, "Show Original")
    apply_action(orig_button, "original")

    # Show the plot
    plt.show()


def plot_mAP_by_giou_with_patch_cli(command_args, prog, description):
    from armory.postprocessing.plot_patch_aware_carla_metric import (
        plot_mAP_by_giou_with_patch,
    )

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
