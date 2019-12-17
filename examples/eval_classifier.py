"""
python -m examples.eval_classifier
"""
from armory.eval import Evaluator
from armory.baseline_models.tf1.simple_keras import SIMPLE_MODEL


if __name__ == "__main__":
    config = {
        "model": SIMPLE_MODEL,
        "attack": "fgsm",
        "data": "mnist",
        "performer": "ta2.twosix",
        "adversarial_knowledge": dict(model="white", defense="aware", data="full"),
        "adversarial_budget": dict(norm="Linf", epsilon="0.3", input_output="inf"),
    }

    rig = Evaluator(config)
    rig.test_classifer()
