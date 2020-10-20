"""
Label-related utilties
"""

import numpy as np


# Targeters assume a numpy 1D array as input to generate


class FixedLabelTargeter:
    def __init__(self, value):
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"value {value} must be a nonnegative int")
        self.value = value

    def generate(self, y):
        return np.ones_like(y) * self.value


class FixedStringTargeter:
    def __init__(self, value):
        if not isinstance(value, str):
            raise ValueError(f"target value {value} is not a string")
        self.value = value

    def generate(self, y):
        return [self.value] * len(y)


class RandomLabelTargeter:
    def __init__(self, num_classes):
        if not isinstance(num_classes, int) or num_classes < 2:
            raise ValueError(f"num_classes {num_classes} must be an int >= 2")
        self.num_classes = num_classes

    def generate(self, y):
        y = y.astype(int)
        return (y + np.random.randint(1, self.num_classes, len(y))) % self.num_classes


class RoundRobinTargeter:
    def __init__(self, num_classes, offset=1):
        if not isinstance(num_classes, int) or num_classes < 1:
            raise ValueError(f"num_classes {num_classes} must be a positive int")
        if not isinstance(offset, int) or offset % num_classes == 0:
            raise ValueError(f"offset {offset} must be an int with % num_classes != 0")
        self.num_classes = num_classes
        self.offset = offset

    def generate(self, y):
        y = y.astype(int)
        return (y + self.offset) % self.num_classes


class ManualTargeter:
    def __init__(self, values, repeat=False):
        if not isinstance(values, list) or not all(isinstance(x, int) for x in values):
            raise ValueError(f'"values" {values} must be a list of ints')
        elif not values:
            raise ValueError('"values" cannot be an empty list')
        self.values = values
        self.repeat = bool(repeat)
        self.current = 0

    def _generate(self, y_i):
        if self.current == len(self.values):
            if self.repeat:
                self.current = 0
            else:
                raise ValueError("Ran out of target labels. Consider repeat=True")

        y_target_i = self.values[self.current]
        self.current += 1
        return y_target_i

    def generate(self, y):
        y_target = []
        for y_i in y:
            y_target.append(self._generate(y_i))
        return np.array(y_target, dtype=int)


class IdentityTargeter:
    def generate(self, y):
        return y.copy().astype(int)
