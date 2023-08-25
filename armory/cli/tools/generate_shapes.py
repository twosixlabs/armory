import argparse
from pathlib import Path

from armory.utils.shape_gen import Shape


def generate_shapes(command_args, prog, description):
    class CustomHelpFormatter(argparse.HelpFormatter):
        def format_help(self):
            help_message = super().format_help()
            replacement = "Available shapes:\n"
            modified_help_message = help_message.replace(
                "positional arguments:", replacement
            )
            search_msg = replacement + "\n"
            search_start = modified_help_message.find(search_msg) + len(search_msg) - 1
            search_end = modified_help_message.find("}", search_start) + 1
            modified_help_message = (
                modified_help_message[:search_start]
                + "\n".join(
                    f"\t{name}" for name in list(Shape._SHAPES.keys()) + ["all"]
                )
                + modified_help_message[search_end:]
            )
            return modified_help_message

    parser = argparse.ArgumentParser(
        prog=prog,
        description=description,
        formatter_class=CustomHelpFormatter,
    )
    parser.add_argument(
        "shape",
        choices=list(Shape._SHAPES.keys()) + ["all"],
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--show", action="store_true", help="Show the generated shape using matplotlib"
    )

    args = parser.parse_args(command_args)

    if args.shape == "all":
        if args.show:
            raise ValueError("Cannot show all shapes")
        for shape_name in Shape._SHAPES:
            generated = Shape.from_name(shape_name)
            if args.output_dir is not None:
                generated.save(args.output_dir / f"{shape_name}.png")
            if args.show:
                generated.show()
    else:
        generated = Shape.from_name(args.shape)
        if args.output_dir is not None:
            generated.save(args.output_dir)
        if args.show:
            generated.show()
