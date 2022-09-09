#!/usr/bin/env false
# -*- coding: utf-8 -*-

"""End-to-end pytest fixtures for testing scenario results.

Example:
    $ pytest --verbose --show-capture=no tests/end_to_end/test_check_results.py

TODO: Decalare this test as being '--no-docker'. Use pytest mark. [woodall]
"""

import json
import math
import pprint
import pytest
import statistics

from pathlib import Path

from armory import paths
from armory.__main__ import run


# Marks all tests in this file as `end_to_end`
pytestmark = pytest.mark.end_to_end

test_data     = Path(Path(__file__).resolve().parent / 'test_check_results/model_scores.json')
host_paths    = paths.runtime_paths()
results_path  = Path(host_paths.output_dir)

pp = pprint.PrettyPrinter(indent=4, width=80, compact=True)


@pytest.fixture(scope="module", autouse=True)
def setup():
    """Pre-flight checks. Future tests will  not continue if setup fails.
    """
    if not all([Path(d).exists() for d in [results_path, test_data]]):
        raise Exception(f"Required application paths do not exist!")



def test_results(capsys):
    scores = {}
    model_results = json.loads(test_data.read_text())

    result_tolerance = 0.05

    for result in results_path.glob("**/*.json"):

        result_json = json.loads(result.read_text())

        results = result_json['results']

        filepath = result_json['config']['sysconfig']['filepath']
        check_used = result_json['config']['sysconfig']['use_gpu']
        gpu_used = result_json['config']['sysconfig']['use_gpu']


        adversarial_scores = results['adversarial_mean_categorical_accuracy']
        benign_scores      = results['benign_mean_categorical_accuracy']


        adversarial_median = statistics.median(adversarial_scores)
        benign_median      = statistics.median(benign_scores)


        with capsys.disabled():
            print("\n\n")
            print(math.isclose(adversarial_median, benign_median, abs_tol = result_tolerance))
            print(all((
                math.isclose(bt, at, abs_tol = result_tolerance) for at, bt in
                    zip(adversarial_scores, benign_scores)
                    if at != 0.0 and bt != 0.0
            )))

        # SETUP
        scores[filepath] = {
            'delta': 0,
            'results': [
                {
                'tolerance': result_tolerance,
                'adversarial_median': adversarial_median,
                'benign_median': benign_median,
                'check_used': check_used,
                'gpu_used': gpu_used
                }
            ]
        }
        # /SETUP

    with capsys.disabled():
        print(json.dumps(scores, indent=2))

