"""
python -m examples.demo

This is an example of running ARMORY evaluation for a simple classifier on MNIST.
The results will be serialized into a JSON file in project root.
"""
import coloredlogs
import logging

from armory.eval import Evaluator

coloredlogs.install(level=logging.INFO)


# TODO: Uncomment these as the fucntionality is added
if __name__ == "__main__":
    config = {
        "performer_name": "ta2.twosix",
        "model_file": "armory.baseline_models.keras.keras_mnist",
        "model_name": "MODEL",
        "eval_file": "performer_evaluation.fgsm_attack",
        "data": "mnist",
        # "performer_repo": None,
        # "adversarial_lib": "ART",
        # "defense": None,
        # "attack": "fgsm",
        # "adversarial_knowledge": dict(model="white", defense="aware", data="full"),
        # "adversarial_budget": dict(norm="Linf", epsilon="0.3", input_output="inf"),
    }
    rig = Evaluator(config)
    rig.run_config()
