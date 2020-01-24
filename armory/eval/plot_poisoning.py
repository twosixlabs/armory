"""
Plot output json files
"""
import json

import logging

logger = logging.getLogger("matplotlib")
logger.setLevel(logging.INFO)

from matplotlib import pyplot as plt


def classification_poisoning(
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
        results = blob["results"]

    name = config["performer_name"]
    data = config["data"]
    knowledge = config["adversarial_knowledge"]["model"]
    defense = config["defense"]

    if output_filepath is None and not show:
        output_filepath = json_filepath
        if output_filepath.endswith(".json"):
            output_filepath = output_filepath[: -len(".json")]
        output_filepath += "_{}.pdf"

    for metric_name in ["backdoor_success_rate", "delta_accuracy"]:
        main_title = (
            f"{name}: {data} for {knowledge}-box attack with {defense} defense."
        )
        percent_poisons, metric = [
            results[x] for x in (["percent_poisons", metric_name])
        ]
        plt.plot(percent_poisons, metric)
        plt.title(main_title)
        plt.xlabel("Fraction of dataset poisoned")
        plt.ylabel(f"Model performance ({metric_name})")
        if show:
            plt.show()
        else:
            plt.tight_layout()
            plt.savefig(output_filepath.format(metric_name), format="pdf")
        plt.close()
