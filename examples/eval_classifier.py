"""
python -m examples.eval_classifier

This is an example of running ARMORY evaluation for a simple classifier on MNIST.
The results will be serialized into a JSON file in project root.
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
        "adversarial_budget": dict(norm="Linf", epsilon="0.3", input_output="inf"),
    }
    rig = Evaluator(config)
    rig.run_config()
