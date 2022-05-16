"""
This module implements the abstract base classes for all extended defenses.
"""

import abc


class Transformer(abc.ABC):
    """
    Abstract base class for model transform defense classes.
    """

    @abc.abstractmethod
    def transform(self, model):
        """
        Returns a transformed version of the target model.
        """
        raise NotImplementedError
