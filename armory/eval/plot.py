"""
Plot output json files
"""
import json
import os

from matplotlib import pyplot as plt
import numpy as np

from armory.logs import log


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


def get_xy(
    benign,
    adversarial,
    epsilon,
    ascending_attack=True,
    min_value=-np.inf,
    max_value=np.inf,
    round_epsilon=False,
):
    """
    ascending_attack - whether the attack gets stronger as epsilon increases
    """
    total = len(benign)
    assert len(benign) == len(adversarial) == len(epsilon)
    assert len(epsilon) > 0
    benign, adversarial, epsilon = (
        np.asarray(x) for x in (benign, adversarial, epsilon)
    )
    ascending_attack = bool(ascending_attack)
    if round_epsilon:
        epsilon = epsilon.round()
    x = []
    y = []

    failed_benign = (benign == 0).sum()
    failed_attack = np.logical_and(benign != 0, adversarial != 0).sum()

    x.append(min_value)
    if ascending_attack:
        y.append(total - failed_benign)
    else:
        y.append(failed_attack)

    index = np.logical_and(benign != 0, adversarial == 0)
    epsilon = epsilon[index]
    for e in sorted(set(epsilon)):
        x.append(e)
        if ascending_attack:
            y.append(total - failed_benign - (epsilon <= e).sum())
        else:
            y.append(total - failed_benign - (epsilon >= e).sum())

    x.add_results(max_value)
    if ascending_attack:
        y.append(failed_attack)
    else:
        y.append(total - failed_benign)

    x, y = np.asarray(x), np.asarray(y)
    y = y / total
    return x, y


class SpeakerID:
    def __init__(self, json_filepaths=None, names=None):
        self.xs = []
        self.ys = []
        self.names = []
        if json_filepaths:
            self.update(json_filepaths, names=names)

    def get_results(self, json_filepath):
        with open(json_filepath) as f:
            blob = json.load(f)

        results = blob["results"]
        adversarial = results["adversarial_categorical_accuracy"]
        benign = results["benign_categorical_accuracy"]
        epsilon = results["perturbation_snr_db"]
        return benign, adversarial, epsilon

    def add(self, json_filepath, name=None):
        if name is None:
            name = os.path.basename(json_filepath)
        benign, adversarial, epsilon = self.get_results(json_filepath)

        x, y = get_xy(
            benign,
            adversarial,
            epsilon,
            ascending_attack=False,
            min_value=0,
            max_value=67,
            round_epsilon=True,
        )
        self.xs.append(x)
        self.ys.append(y)
        self.names.append(name)

    def update(self, json_filepaths, names=None):
        if names is None:
            names = [None] * len(json_filepaths)

        for json_filepath, name in zip(json_filepaths, names):
            self.add(json_filepath, name=name)

    def plot(
        self,
        save_filepath=None,
        show=True,
        title="Speaker ID Classification, L2 SNR Attack",
        xlabel="SNR (dB)",
        ylabel="Task Accuracy",
    ):
        if save_filepath is not None and show is True:
            log.warning("setting show to False")
            show = False

        for x, y, name in zip(self.xs, self.ys, self.names):
            x = np.array([-3] + list(x))
            y = np.array([0] + list(y))
            plt.plot(x, y, "x-", label=name)

        plt.xlabel(xlabel)
        plt.xticks(
            ticks=[-3] + list(np.arange(0, 64, 10)) + [64, 67],
            labels=[-3]
            + list(str(x) for x in np.arange(0, 64, 10))
            + [64, r"$\infty$"],
        )
        plt.ylabel(ylabel)
        plt.yticks(ticks=np.arange(0, 1.01, 0.2))
        plt.title(title)
        plt.legend(loc="upper left")
        plt.axis([0 - 5, 64 + 5, 0 - 0.05, 1 + 0.05])

        if show:
            plt.show()
        else:
            plt.tight_layout()
            plt.savefig(save_filepath, format="pdf")
        plt.close()


class SpeakerIDLinf:
    def __init__(self, json_filepaths=None, names=None):
        self.xs = []
        self.ys = []
        self.names = []
        if json_filepaths:
            self.update(json_filepaths, names=names)

    def get_results(self, json_filepath):
        with open(json_filepath) as f:
            blob = json.load(f)

        results = blob["results"]
        adversarial = results["adversarial_categorical_accuracy"]
        benign = results["benign_categorical_accuracy"]
        epsilon = results["perturbation_linf"]
        return benign, adversarial, epsilon

    def add(self, json_filepath, name=None):
        if name is None:
            name = os.path.basename(json_filepath)
        benign, adversarial, epsilon = self.get_results(json_filepath)

        print(min(epsilon), max(epsilon))
        x, y = get_xy(
            benign,
            adversarial,
            epsilon,
            ascending_attack=True,
            min_value=0,
            max_value=0.06,
            round_epsilon=False,
        )
        print(x, y)
        self.xs.append(x)
        self.ys.append(y)
        self.names.append(name)

    def update(self, json_filepaths, names=None):
        if names is None:
            names = [None] * len(json_filepaths)

        for json_filepath, name in zip(json_filepaths, names):
            self.add(json_filepath, name=name)

    def plot(
        self,
        save_filepath=None,
        show=True,
        title=r"Speaker ID Classification, L$\infty$ Attack",
        xlabel=r"Epsilon (L$\infty$)",
        ylabel="Task Accuracy",
    ):
        if save_filepath is not None and show is True:
            log.warning("setting show to False")
            show = False

        for x, y, name in zip(self.xs, self.ys, self.names):
            x = np.array(list(x))
            y = np.array(list(y))
            plt.plot(x, y, "x-", label=name)

        plt.xlabel(xlabel)
        plt.xticks(ticks=np.arange(0, 0.0201, 0.002))
        #    ticks=[-3] + list(np.arange(0, 64, 10)) + [64, 67],
        #    labels=[-3]
        #    + list(str(x) for x in np.arange(0, 64, 10))
        #    + [64, r"$\infty$"],
        #
        plt.ylabel(ylabel)
        # plt.yticks(ticks=np.arange(0, 1.01, 0.2))
        plt.title(title)
        plt.legend(loc="upper right")
        plt.axis([0 - 0.001, 0.02 + 0.001, 0 - 0.05, 1 + 0.05])

        if show:
            plt.show()
        else:
            plt.tight_layout()
            plt.savefig(save_filepath, format="pdf")
        plt.close()
