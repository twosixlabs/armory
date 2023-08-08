"""
Utility functions for visualizing the output of the metric "object_detection_AP_per_class_by_giou_from_patch."
This metric captures how adversarial AP varies as objects get farther from the patch.

There are two functions which produce slightly different plots:
plot_mAP_by_giou_with_patch, and plot_single_giou_threshold.

plot_mAP_by_giou_with_patch() shows how the mAP changes over a range of GIoU values.  It
can display this in three "flavors": cumulative by max GIoU, cumulative by min GIoU,
or as a histogram.  The first flavor represents all objects outside each given range of GIoU
(i.e. further from the patch, or more negative GIoU).  The second flavor is all objects within
each range (i.e. closer to the patch, or more positive GIoU).  The histogram version reports AP
for disjoint intervals of GIoU value.

plot_single_giou_threshold() only considers one user-specified GIoU threshold, and shows
the adversarial and benign AP both above and below that threshold.  For example, using a threshold
of 0.0 would correspond to all objects touching and not touching the patch.

Intended for stand-alone usage:
>>> from armory.postprocessing.plot_patch_aware_carla_metric import plot_mAP_by_giou_with_patch
>>> plot_mAP_by_giou_with_patch("path/to/results.json", flavors=["cumulative_by_min_giou"])

"""
import json

from matplotlib import pyplot as plt
import numpy as np

ben_color = "tab:blue"
adv_color = "tab:red"

flavor_titles = {
    "cumulative_by_max_giou": "Cumulative by Upper Bound\n(Exclusive)",
    "cumulative_by_min_giou": "Cumulative by Lower Bound\n(Inclusive)",
    "histogram_left": "Histogram\n(Left Oriented)",
}

fontsize = 8


def _init_plots(json_filepath, include_classes, n_flavors=1):
    # Read json from json_filepath and initialize fig and ax objects for plotting.

    # Returns giou dicts from json, the fig and axes, as well as number of classes.

    with open(json_filepath) as f:
        blob = json.load(f)
        results = blob["results"]

    if "adversarial_object_detection_AP_per_class_by_giou_from_patch" not in results:
        raise ValueError(
            "Provided json does not have results.adversarial_object_detection_AP_per_class_by_giou_from_patch."
        )
    adv_giou = results["adversarial_object_detection_AP_per_class_by_giou_from_patch"][
        0
    ]
    ben_giou = results["benign_object_detection_AP_per_class_by_giou_from_patch"][0]
    adv_ap = results["adversarial_carla_od_AP_per_class"][0]["mean"]
    ben_ap = results["benign_carla_od_AP_per_class"][0]["mean"]

    if include_classes:
        n_class = len(ben_giou["cumulative_by_max_giou"]["0.0"]["class"])
    else:
        n_class = 0

    fig, axes = plt.subplots(n_class + 1, n_flavors, sharex=True, sharey=True)
    # Ensure axes are in an array for consistent reference
    if not isinstance(axes, np.ndarray) or len(axes.shape) == 1:
        axes = np.array(axes).reshape(n_class + 1, n_flavors)

    axes[0, 0].set_ylabel("Mean AP")

    for ax in axes.flatten():
        # Add dotted lines for total mAP
        ax.set_ylim(0, 1.05)
        ax.axhline(
            y=ben_ap,
            linestyle="dotted",
            color=ben_color,
            linewidth=1.5,
            label="Ben. mAP: {0:.2f}".format(ben_ap),
        )
        ax.axhline(
            y=adv_ap,
            linestyle="dotted",
            color=adv_color,
            linewidth=1.5,
            label="Adv. mAP: {0:.2f}".format(adv_ap),
        )

    return ben_giou, adv_giou, n_class, (fig, axes)


def plot_mAP_by_giou_with_patch(
    json_filepath, flavors=None, show=True, output_filepath=None, include_classes=True
):
    """Plots mAP of boxes according to their giou with the patch.

    json_filepath: the path to the output json file.
    flavors: a list of data accumulation variants.
            Subset of ["cumulative_by_max_giou", "cumulative_by_min_giou", "histogram_left"]
            None defaults to all flavors.
    show: whether to show the plot.
    output_filepath: if provided, figure is saved.
    include_classes: include subplots for each class.
    """

    if flavors is None:
        flavors, titles = zip(*flavor_titles.items())
    else:
        for flavor in flavors:
            if flavor not in flavor_titles:
                raise ValueError(
                    f"Invalid flavor {flavor}; should be one of {list(flavor_titles.keys())}"
                )
        titles = [flavor_titles[flavor] for flavor in flavors]

    ben_giou, adv_giou, n_class, fig_ax = _init_plots(
        json_filepath, include_classes, n_flavors=len(flavors)
    )
    fig, axes = fig_ax

    fig.suptitle("AP by GIoU with patch")
    for i in range(len(titles)):
        axes[0, i].set_title(titles[i])
        axes[-1, i].set_xlabel("GIoU")

    def add_bars(
        ax,
        x,
        y,
        offsets=[0, 0.04],
        colors=[ben_color, adv_color],
        labels=["Benign", "Adversarial"],
    ):
        # Helper function to plot bars.  x and y each contain benign and adversarial data for looping over
        width = 0.04
        for x_, y_, o, c, l in zip(x, y, offsets, colors, labels):
            rects = ax.bar(x_ + o, y_, width, color=c, label=l)
            if len(flavors) == 1:
                # If figure is only 1 subplot wide, annotate bars with their values.
                # Too messy for figures with more subplots.
                ax.bar_label(rects, fmt="%.2f", label_type="center", fontsize=fontsize)

    for i, flavor in enumerate(flavors):
        # Get data and add plots for each flavor.

        ben = ben_giou[flavor]
        adv = adv_giou[flavor]
        m_b = np.array([[float(k), ben[k]["mean"]] for k in ben])
        m_a = np.array([[float(k), adv[k]["mean"]] for k in adv])

        # Plot mean data
        add_bars(axes[0, i], (m_b[:, 0], m_a[:, 0]), (m_b[:, 1], m_a[:, 1]))

        for j in range(1, n_class + 1):
            # Plot per-class data
            m_b = np.array([[float(k), ben[k]["class"][str(j)]] for k in ben])
            m_a = np.array([[float(k), adv[k]["class"][str(j)]] for k in adv])
            add_bars(axes[j, i], (m_b[:, 0], m_a[:, 0]), (m_b[:, 1], m_a[:, 1]))

            axes[j, 0].set_ylabel(f"Class {j} AP")

    # Set up legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc=8, ncol=4)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.15)

    if output_filepath is not None:
        plt.savefig(output_filepath)
    if show:
        plt.show()


def plot_single_giou_threshold(
    json_filepath, threshold=0.0, show=True, output_filepath=None, include_classes=True
):
    """Plots mAP of boxes over and below a given GIoU threshold.

    json_filepath: the path to the output json file.
    threshold: the threshold of interest.  Must be a valid key in the json results.
    show: whether to show the plot.
    output_filepath: if provided, figure is saved.
    include_classes: include subplots for each class.
    """

    ben_giou, adv_giou, n_class, fig_ax = _init_plots(json_filepath, include_classes)
    fig, axes = fig_ax

    ap_dicts = [
        ben_giou["cumulative_by_max_giou"][str(threshold)],
        ben_giou["cumulative_by_min_giou"][str(threshold)],
        adv_giou["cumulative_by_max_giou"][str(threshold)],
        adv_giou["cumulative_by_min_giou"][str(threshold)],
    ]

    fig.suptitle(f"AP by GIoU with patch\nrelative to threshold of {threshold}")

    def add_bars(
        ax, d_list, colors=[ben_color, adv_color], labels=["Benign", "Adversarial"]
    ):
        # Helper function to plot bars.  d_list contains benign and adversarial data for looping over
        x = np.arange(2)
        width = 0.25
        multiplier = -0.5

        for d, c, l in zip(d_list, colors, labels):
            offset = width * multiplier + width
            rects = ax.bar(x + offset, d, width, label=l, color=c)
            ax.bar_label(rects, fmt="%.2f", label_type="center", fontsize=fontsize)
            multiplier *= -1

        ax.set_xticks(x + width)
        ax.set_xticklabels(["Below threshold", "Above threshold"])

    # Plot mean data
    add_bars(axes[0, 0], np.array([d["mean"] for d in ap_dicts]).reshape(2, 2))

    for j in range(1, n_class + 1):
        # Plot per-class data
        add_bars(
            axes[j, 0], np.array([d["class"][str(j)] for d in ap_dicts]).reshape(2, 2)
        )
        axes[j, 0].set_ylabel(f"Class {j} AP")

    # Set up legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc=8, ncol=4)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.15)

    if output_filepath is not None:
        plt.savefig(output_filepath)
    if show:
        plt.show()
