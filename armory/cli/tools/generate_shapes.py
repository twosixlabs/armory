import argparse
from pathlib import Path

from armory.utils.shape_gen import Shape


def generate_shapes(command_args, prog, description):
    parser = argparse.ArgumentParser(
        prog=prog,
        description=description,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "shape",
        choices=list(Shape._SHAPES.keys()) + ["all"],
        help="Shape to generate or 'all' to generate all shapes",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--show", action="store_true")

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
            generated.save(args.output_dir / f"{args.shape}.png")
        if args.show:
            generated.show()
