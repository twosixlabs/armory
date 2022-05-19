"""
Plot output json files
"""
import json

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

    data = config["dataset"]["name"]
    knowledge = config["attack"]["knowledge"]
    if config["defense"]:
        defense = config["defense"]["name"]

    if output_filepath is None and not show:
        output_filepath = json_filepath
        if output_filepath.endswith(".json"):
            output_filepath = output_filepath[: -len(".json")]
        output_filepath += "_{}.pdf"

    for metric_name in [
        "undefended_backdoor_success_rate",
        "defended_backdoor_success_rate",
        "delta_accuracy",
    ]:
        main_title = f"{data} for {knowledge}-box attack \nwith {defense} defense."
        fraction_poisons = results[metric_name + "_mean"].keys()
        metric_mean = [results[metric_name + "_mean"][k] for k in fraction_poisons]
        metric_std = [results[metric_name + "_std"][k] for k in fraction_poisons]
        fraction_poisons = list(map(float, fraction_poisons))

        plt.errorbar(fraction_poisons, metric_mean, metric_std, capsize=5)
        plt.title(main_title)
        plt.xlabel("Fraction of dataset poisoned")
        plt.ylabel(f"Model performance ({metric_name})")
        if show:
            plt.show()
        else:
            plt.tight_layout()
            plt.savefig(output_filepath.format(metric_name), format="pdf")
        plt.close()
