"""
python -m examples.eval_classifier_all_delta

Runs ARMORY evaluation on basic MNIST model across all epsilon/delta values
    Uses L1, L2, and Linf norms
"""

import logging, coloredlogs
import json

coloredlogs.install(level=logging.DEBUG)

from armory.eval import Evaluator

if __name__ == "__main__":
    with open("examples/mnist_config.json") as f:
        config = json.load(f)
    rig = Evaluator(config)
    rig.run_config()
