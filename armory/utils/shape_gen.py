# To add a new shape, add its name and function to _SHAPES
from dataclasses import dataclass, field
import inspect
from io import BytesIO
import os
from pathlib import Path
from typing import Union

from PIL import Image
import cairosvg
import matplotlib.pyplot as plt
import numpy as np
import svg

RADIUS_INNER = -50  # 25
STROKE_WIDTH = 2


def _write(outdir: Union[str, Path, BytesIO], **kwargs):
    if not isinstance(outdir, BytesIO):
        outfile = os.path.join(
            os.path.abspath(outdir),
            f"{inspect.currentframe().f_back.f_code.co_name}.png",
        )
    else:
        outfile = outdir
    plt.savefig(outfile, transparent=True, **kwargs)
    plt.close()
    return outfile


def asterisk(outdir: Union[str, Path, BytesIO], **kwarg_catcher) -> Union[str, BytesIO]:
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
    return _write(outdir)


def circle(outdir: Union[str, Path, BytesIO], **kwarg_catcher) -> Union[str, BytesIO]:
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
    return _write(outdir)


def grid(
    outdir: Union[str, Path, BytesIO],
    num_circles=25,
    circle_radius=0.5,
    circle_spacing=2.5,
    **kwarg_catcher,
) -> Union[str, BytesIO]:
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

    # Create grid of circles
    for i in range(num_circles):
        for j in range(num_circles):
            # Calculate center of circle
            x = circle_radius + j * circle_spacing
            y = circle_radius + i * circle_spacing

            # Add circle to plot
            circle = plt.Circle((x, y), circle_radius, color="blue", edgecolor="none")
            ax.add_artist(circle)

    # Save figure as png with transparent background
    return _write(outdir, bbox_inches="tight", pad_inches=0)


# Start of SVG generation code
def _write_svg(
    shape: svg.SVG, outdir: Union[str, Path, BytesIO]
) -> Union[str, BytesIO]:
    if not isinstance(outdir, BytesIO):
        outfile = os.path.join(
            os.path.abspath(outdir),
            f"{inspect.currentframe().f_back.f_code.co_name}.png",
        )
    else:
        outfile = outdir
    cairosvg.svg2png(bytestring=str(shape), write_to=outfile)
    return outfile


def _circles_outer(num_rot, cent_x, cent_y, rad):
    circles_lst = []
    angle = 360 / num_rot
    curr = 0
    for i in range(num_rot):
        curr = i * angle
        circles_lst.append(_circle_outer(curr, cent_x, cent_y, rad))
    return circles_lst


def _circle_outer(rotation, cent_x, cent_y, rad):
    return svg.Circle(
        cx=cent_x,
        cy=cent_y - rad - RADIUS_INNER,
        r=rad,
        stroke="#F73859",
        fill="transparent",
        stroke_width=STROKE_WIDTH,
        transform=[
            svg.Rotate(
                rotation, cent_x, cent_y
            )  # rotate wrt shape, 90=rotational value, x, y
        ],
    )


def _circles_concentric(num_it, cent_x, cent_y, min_rad, max_rad):
    circles_lst = []
    if num_it == 0:
        return []
    rad_increase = (max_rad - min_rad) / num_it  # (outer_rad - inner_rad) / num_it
    rad = min_rad
    while rad <= max_rad:
        circles_lst.append(_circle_concentric(cent_x, cent_y, rad))
        rad += rad_increase
    return circles_lst


def _circle_concentric(cent_x, cent_y, rad):
    return svg.Circle(
        cx=cent_x,
        cy=cent_y,
        r=rad,
        stroke="black",
        fill="transparent",
        stroke_width=STROKE_WIDTH,
    )


def concentric_circles(
    outdir: Union[str, Path, BytesIO],
    num_it=0,
    cent_x=250,
    cent_y=250,
    min_rad=RADIUS_INNER,
    max_rad=200,
    # circles outer
    num_rot=20,
    rad=100,
    width=500,
    height=500,
    **kwarg_catcher,
) -> Union[str, BytesIO]:
    if not isinstance(outdir, BytesIO) and not os.path.isdir(outdir):
        raise ValueError(f"outdir must be valid a directory or BytesIO, got: {outdir}")
    circles_con = _circles_concentric(num_it, cent_x, cent_y, min_rad, max_rad)
    circles_out = _circles_outer(num_rot, cent_x, cent_y, rad)

    out = svg.SVG(width=width, height=height, elements=circles_con + circles_out)

    _write_svg(out, outdir)


def _kites(num_rot, cent_x, cent_y):
    kites_lst = []
    angle = 360 / num_rot
    curr = 0
    for i in range(num_rot):
        curr = i * angle
        kites_lst.append(_kite(curr, cent_x, cent_y))
    return kites_lst


def _kite(rotation, cent_x, cent_y):
    return svg.Polygon(
        points=[
            250,
            250,  # center
            215,
            100,  # left point
            250,
            80,  # center top point (origin - (diff between radii))
            285,
            100,  # right point
        ],
        stroke="#F73859",
        fill="#F73859",  # "transparent", #181a1b
        stroke_width=STROKE_WIDTH,
        transform=[
            svg.Rotate(
                rotation, cent_x, cent_y
            )  # rotate wrt shape, 90=rotational value, x, y
        ],
    )


def _kite_lines(num_rot, x1, y1, x2, y2):
    lines_lst = []
    angle = 360 / num_rot
    curr = 0
    for i in range(num_rot):
        curr = i * angle
        lines_lst.append(_kite_line(curr, x1, y1, x2, y2))
    return lines_lst


def _kite_line(rotation, cx, cy, ox, oy):
    return svg.Line(
        x1=cx,
        y1=cy,
        x2=ox,
        y2=oy,
        stroke="#F73859",
        stroke_width=STROKE_WIDTH,
        transform=[
            svg.Rotate(rotation, cx, cy)  # rotate wrt shape, 90=rotational value, x, y
        ],
    )


def jxcr_gear(
    outdir: Union[str, Path, BytesIO], **kwarg_catcher
) -> Union[str, BytesIO]:
    kite_elems = _kites(8, 250, 250)
    kite_line_elems = _kite_lines(
        8, 250, 250, 250, 50
    )  # num_rotations, center_x, center_y, outer_x (center_x - outer_radius), outer_y (center_y - outer_radius)
    circle_elems = [
        svg.Circle(
            cx=250,
            cy=250,
            r=200,
            stroke="#F73859",
            fill="transparent",
            stroke_width=STROKE_WIDTH,
        ),
        svg.Circle(
            cx=250,
            cy=250,
            r=125,
            stroke="#F73859",
            fill="transparent",
            stroke_width=STROKE_WIDTH,
        ),
        svg.Circle(
            cx=250,
            cy=250,
            r=25,
            stroke="#F73859",
            fill="#F73859",
            stroke_width=2,
        ),
    ]
    return _write_svg(
        shape=svg.SVG(
            width=500, height=500, elements=kite_line_elems + kite_elems + circle_elems
        ),
        outdir=outdir,
    )


# Grid-like
def _sierpinski(n, p1, p2, p3):
    """
    Recursively generates a Sierpinski triangle.
    """
    if n == 0:
        return [p1, p2, p3]
    else:
        p12 = (p1 + p2) / 2
        p23 = (p2 + p3) / 2
        p31 = (p3 + p1) / 2
        s1 = _sierpinski(n - 1, p1, p12, p31)
        s2 = _sierpinski(n - 1, p12, p2, p23)
        s3 = _sierpinski(n - 1, p31, p23, p3)
        return np.vstack((s1, s2, s3))


def sierpinski(
    outdir: Union[str, Path, BytesIO], **kwarg_catcher
) -> Union[str, BytesIO]:
    depth = 5  # depth of recursion
    # Triangle vertices
    p1 = np.array([0, 0])
    p2 = np.array([1, 0])
    p3 = np.array([0.5, np.sqrt(3) / 2])

    points = _sierpinski(depth, p1, p2, p3)

    # fig, ax = plt.subplots()
    fig = plt.figure(frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])

    ax.scatter(points[:, 0], points[:, 1], s=1, color="darkgray")

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_aspect("equal")

    ax.set_xticks([])
    ax.set_yticks([])

    # ax.set_title('Sierpinski Triangle')
    ax.set_axis_off()
    ax.set_facecolor((1, 1, 1, 0))
    return _write(outdir, bbox_inches="tight", pad_inches=0)


# def donut(
#         outdir: Union[str, Path, BytesIO],
#         **kwarg_catcher,
# ):
#     if not isinstance(outdir, BytesIO) and not os.path.isdir(outdir):
#         raise ValueError(f"outdir must be valid a directory, got: {outdir}")
#     def create_donut(cx, cy, outer_radius, inner_radius):
#         donut_group = svg.Group()

#         outer_circle = svg.shapes.Circle(
#             center=(cx, cy),
#             r=outer_radius,
#             stroke="#F73859",
#             fill="transparent",
#             stroke_width=STROKE_WIDTH,
#         )
#         donut_group.add(outer_circle)

#         inner_circle = svg.shapes.Circle(
#             center=(cx, cy),
#             r=inner_radius,
#             stroke="#F73859",
#             fill="transparent",
#             stroke_width=STROKE_WIDTH,
#         )
#         donut_group.add(inner_circle)

#         return donut_group

#     return _write_svg(
#         shape=svg.SVG(
#             size=(500, 500),
#             elements=[
#                 create_donut(250, 250, 200, 125),
#                 svg.shapes.Circle(
#                     center=(250, 250),
#                     r=25,
#                     stroke="#F73859",
#                     fill="#F73859",
#                     stroke_width=2,
#                 ),
#             ],
#         ),
#         outdir=outdir,
#     )


@dataclass
class Shape:
    func: callable
    kwargs: dict = field(default_factory=dict)
    _SHAPES = {
        f.__name__: f
        for f in (asterisk, circle, grid, concentric_circles, jxcr_gear, sierpinski)
    }

    @classmethod
    def from_name(cls, name, **kwargs) -> "Shape":
        if name is None:
            return
        if name not in cls._SHAPES:
            raise ValueError(
                f"Invalid shape name: {name}. Must be one of: {cls._SHAPES.keys()}"
            )
        return cls(func=cls._SHAPES[name], **kwargs)

    @property
    def array(self):
        with BytesIO() as f:
            self.func(f, **self.kwargs)
            f.seek(0)
            return np.array(Image.open(f))

    @property
    def name(self):
        if self.func.__name__ in self._SHAPES:
            return self.func.__name__
        candidates = [k for k, v in self._SHAPES.items() if v == self.func]
        if len(candidates) == 1:
            return candidates[0]
        raise ValueError(f"Could not find name for shape: {self.func}")

    def save(self, outdir=os.path.join(os.path.dirname(__file__), "masks")):
        return self.func(outdir, **self.kwargs)

    def show(self):
        plt.imshow(self.array)
        plt.show()
