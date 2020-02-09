"""
Plot output json files
"""
import json

from matplotlib import pyplot as plt


def classification(
    json_filepath="outputs/latest.json", output_filepath=None, show=False
):
    """
    Plot classification results

    json_filepath - filepath for json file
    output_filepath - filepath for saving output graph
        if None, use json_filepath and change ending to .pdf
    show - if True, show the plot instead of saving to file
    """
    with open(json_filepath) as f:
        blob = json.load(f)
        config = blob["config"]
        all_results = blob["results"]

    data = config["dataset"]["name"]
    knowledge = config["attack"]["knowledge"]
    defense = config["defense"]

    if output_filepath is None and not show:
        output_filepath = json_filepath
        if output_filepath.endswith(".json"):
            output_filepath = output_filepath[: -len(".json")]
        output_filepath += "_{}.pdf"

    main_title = f"{data} for {knowledge}-box attack with {defense} defense."
    for norm, results in all_results.items():
        mapping = {
            "L0": r"$L_0$",
            "L1": r"$L_1$",
            "L2": r"$L_2$",
            "Lp": r"$L_p$",
            "Linf": r"$L_\infty$",
        }
        norm_fancy = mapping[norm]

        epsilons, metric, values = [
            results[x] for x in ("epsilons", "metric", "values")
        ]
        plt.plot(epsilons, values)
        plt.title(main_title)
        plt.xlabel(f"{norm_fancy} attack strength for normalized input")
        plt.ylabel(f"Model performance ({metric})")
        if show:
            plt.show()
        else:
            plt.tight_layout()
            plt.savefig(output_filepath.format(norm), format="pdf")
        plt.close()
