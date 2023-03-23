# To add a new shape, add its name and function to _SHAPES
from dataclasses import dataclass
import inspect
from io import BytesIO
import os

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def asterisk(outdir) -> str:
    if not isinstance(outdir, BytesIO) and not os.path.isdir(outdir):
        raise ValueError(f"outdir must be valid a directory or BytesIO, got: {outdir}")
    theta = np.linspace(0, 2 * np.pi, 18, endpoint=False)
    r = np.zeros_like(theta)
    r[::2] = 1
    r[1::2] = 0.5
    # r[::-1] = 1
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    y1 = np.append(y, y[0])
    x1 = np.append(x, x[0])
    fig, ax = plt.subplots()
    ax.plot(x1, y1, "-o", color="slateblue")
    ax.set_axis_off()
    ax.fill(x1, y1, color="slateblue")
    ax.set_facecolor((1, 1, 1, 0))
    if not isinstance(outdir, BytesIO):
        outfile = os.path.join(
            os.path.abspath(outdir), f"{inspect.currentframe().f_code.co_name}.png"
        )
    else:
        outfile = outdir
    fig.savefig(outfile, transparent=True)
    return outfile


def circle(outdir) -> str:
    if not isinstance(outdir, BytesIO) and not os.path.isdir(outdir):
        raise ValueError(f"outdir must be valid a directory, got: {outdir}")
    radius = 1
    num_points = 100

    # Generate the angles for the points on the circle
    theta = np.linspace(0, 2 * np.pi, num_points)

    # Calculate the coordinates of the points on the circle
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    fig, ax = plt.subplots()

    a = 0.5  # adjust opacity
    ax.plot(x, y, "slateblue", alpha=a)

    # ax.set_xlim(-1, 1)
    # ax.set_ylim(-1, 1)
    ax.set_aspect("equal")

    ax.set_axis_off()
    ax.fill(x, y, color="slateblue", alpha=a)
    # ax.set_facecolor((1, 1, 1, 0))
    ax.margins(0, 0)
    if not isinstance(outdir, BytesIO):
        outfile = os.path.join(
            os.path.abspath(outdir), f"{inspect.currentframe().f_code.co_name}.png"
        )
    else:
        outfile = outdir
    fig.savefig(outfile, transparent=True, bbox_inches="tight", pad_inches=0)
    return outfile


def grid(outdir, num_circles=25, circle_radius=0.5, circle_spacing=2.5) -> str:
    if not isinstance(outdir, BytesIO) and not os.path.isdir(outdir):
        raise ValueError(f"outdir must be valid a directory, got: {outdir}")
    # Set up figure and axes
    fig, ax = plt.subplots(figsize=(10, 10))

    # Set background color to transparent
    fig.patch.set_alpha(0.0)

    # Set aspect ratio to be equal so circles are perfectly round
    ax.set_aspect("equal")
    ax.set_axis_off()

    # Calculate circle radius and spacing between circles
    circle_spacing = circle_radius * circle_spacing

    # Set x and y limits based on circle_radius and spacing
    ax.set_xlim(
        0,
        circle_radius * 2 * num_circles
        + (circle_spacing - circle_radius * 2) * (num_circles - 1),
    )
    ax.set_ylim(
        0,
        circle_radius * 2 * num_circles
        + (circle_spacing - circle_radius * 2) * (num_circles - 1),
    )

    # Create 10x10 grid of circles
    for i in range(num_circles):
        for j in range(num_circles):
            # Calculate center of circle
            x = circle_radius + j * circle_spacing
            y = circle_radius + i * circle_spacing

            # Add circle to plot
            circle = plt.Circle((x, y), circle_radius, color="blue", edgecolor="none")
            ax.add_artist(circle)

    # Save figure as png with transparent background
    if not isinstance(outdir, BytesIO):
        outfile = os.path.join(
            os.path.abspath(outdir), f"{inspect.currentframe().f_code.co_name}.png"
        )
    else:
        outfile = outdir
    plt.savefig(outfile, bbox_inches="tight", pad_inches=0, transparent=True)
    return outfile


_SHAPES = {f.__name__: f for f in (asterisk, circle, grid)}


@dataclass
class Shape:
    func: callable

    @classmethod
    def from_name(cls, name) -> "Shape":
        if name is None:
            return
        if name not in _SHAPES:
            raise ValueError(
                f"Invalid shape name: {name}. Must be one of: {_SHAPES.keys()}"
            )
        return cls(_SHAPES[name])

    @property
    def array(self):
        with BytesIO() as f:
            self.func(f)
            f.seek(0)
            return np.array(Image.open(f))

    @property
    def name(self):
        if self.func.__name__ in _SHAPES:
            return self.func.__name__
        candidates = [k for k, v in _SHAPES.items() if v == self.func]
        if len(candidates) == 1:
            return candidates[0]
        raise ValueError(f"Could not find name for shape: {self.func}")

    def save(self, outdir=os.path.join(os.path.dirname(__file__), "masks")):
        return self.func(outdir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Generate various png masks")
    parser.add_argument("shape", choices=_SHAPES.keys() + ["all"])
    args = parser.parse_args()

    if args.shape == "all":
        for shape in _SHAPES:
            shape()
    else:
        for shape in _SHAPES:
            if shape.__name__ == args.shape:
                shape()
                break
