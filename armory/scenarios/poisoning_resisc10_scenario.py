"""
Classifier evaluation within ARMORY

Scenario Contributor: MITRE Corporation
"""

from armory.scenarios.poison import Poison


class RESISC10(Poison):
    """
    Dirty label poisoning on resisc10 dataset

    NOTE: "validation" is a split for resisc10 that is currently unused
    """
