"""
python run_evaluation.py <json_config>

Try: 
python run_evaluation.py examples/mnist_fgsm_all_epsilon.json

This runs an arbitrary config file. Results are output to the `outputs/` directory.
"""

import argparse
import json
import logging

import coloredlogs

from armory.eval import Evaluator


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Armory from config file.")
    parser.add_argument(
        "filepath", metavar="<json_config>", type=str, help="json config file"
    )
    parser.add_argument(
        "-d",
        "--debug",
        dest="log_level",
        action="store_const",
        const=logging.DEBUG,
        default=logging.INFO,
        help="Debug output (logging=DEBUG)",
    )
    args = parser.parse_args()

    coloredlogs.install(level=args.log_level)
    with open(args.filepath) as f:
        config = json.load(f)
    rig = Evaluator(config)
    rig.run_config()
