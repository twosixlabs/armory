"""
python -m examples.eval_classifier_all_delta

Runs ARMORY evaluation on basic MNIST model across all epsilon/delta values
    Uses L1, L2, and Linf norms
"""

import logging, coloredlogs
coloredlogs.install(level=logging.INFO)

from armory.eval import Evaluator

if __name__ == "__main__":
    config = {
        "model_wrapper": "ART",
        "model_file": "armory.baseline_models.tf1.simple_keras",
        "model_name": "SIMPLE_MODEL",
        "defense": None,
        "attack": "fgsm",
        "data": "mnist",
        "performer_name": "ta2.twosix",
        "performer_repo": None,
        "adversarial_knowledge": dict(model="white", defense="aware", data="full"),
        "adversarial_budget": dict(norm=["L1", "L2", "Linf"], epsilon="all", input_output="inf"),
    }
    rig = Evaluator(config)
    rig.run_config()
