"""
Label-related utilties
"""

import importlib

import numpy as np

# Targeters assume a numpy 1D array as input to generate


def import_from_module(name, attribute):
    if not isinstance(name, str) or not isinstance(attribute, str):
        raise ValueError(
            "When 'import_from' is used, it and the attribute must be of type str,"
            f" not {name} and {attribute}"
        )
    module = importlib.import_module(name)
    return getattr(module, attribute)


class FixedLabelTargeter:
    def __init__(self, *, value):
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"value {value} must be a nonnegative int")
        self.value = value

    def generate(self, y):
        return np.ones_like(y) * self.value


class FixedStringTargeter:
    def __init__(self, *, value):
        if not isinstance(value, str):
            raise ValueError(f"target value {value} is not a string")
        self.value = value

    def generate(self, y):
        return [self.value] * len(y)


class RandomLabelTargeter:
    def __init__(self, *, num_classes):
        if not isinstance(num_classes, int) or num_classes < 2:
            raise ValueError(f"num_classes {num_classes} must be an int >= 2")
        self.num_classes = num_classes

    def generate(self, y):
        y = y.astype(int)
        return (y + np.random.randint(1, self.num_classes, len(y))) % self.num_classes


class RoundRobinTargeter:
    def __init__(self, *, num_classes, offset=1):
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
    def __init__(self, *, values, import_from=False, repeat=False, dtype=int):
        if import_from:
            values = import_from_module(import_from, values)
        if not values:
            raise ValueError('"values" cannot be an empty list')
        self.values = values
        self.repeat = bool(repeat)
        self.current = 0
        self.dtype = dtype

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
        return np.array(y_target, dtype=self.dtype)


class IdentityTargeter:
    def generate(self, y):
        return y.copy().astype(int)


class ObjectDetectionFixedLabelTargeter:
    """
    Replaces the ground truth labels with the specified value. Does not modify
    the number of boxes or location of boxes.
    """

    def __init__(self, *, value, score=1.0):
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"value {value} must be a nonnegative int")
        self.value = value
        self.score = score

    def generate(self, y):
        targeted_y = []
        for y_i in y:
            target_y_i = y_i.copy()
            target_y_i["labels"] = (
                np.ones_like(y_i["labels"]).reshape((-1,)) * self.value
            )
            target_y_i["scores"] = (
                np.ones_like(y_i["labels"]).reshape((-1,)) * self.score
            ).astype(np.float32)
            target_y_i["boxes"] = y_i["boxes"].reshape((-1, 4))
            targeted_y.append(target_y_i)
        return targeted_y


class CARLAOverObjectDetectionRandomTargeter:
    """
    Generate random annotations of CARLA objects, specifically pedestrians and vehicles,
    using known statistics from the CARLA overhead dataset
    """

    def __init__(self, *, hallucination_per_label=[100, 100]):
        # Object statistics from the CARLA Eval 6 overhead training data
        self.hallucination_labels = [1, 2]  # [pedestrian, vehicle]
        self.hallucination_slopes = [0.27, 0.43]
        self.hallucination_intercepts = [18.0, 40.0]
        self.hallucination_min_widths = [6.0, 6.0]
        self.hallucination_max_widths = [81.0, 277.0]
        self.hallucination_width_means = [25.5, 56.3]
        self.hallucination_width_stds = [13.0, 32.9]
        self.hallucination_per_label = hallucination_per_label
        self.X_MAX = 1280  # input resolution
        self.Y_MAX = 960

        if isinstance(hallucination_per_label, list):
            if len(hallucination_per_label) != len(self.hallucination_labels):
                raise ValueError(
                    f"hallucination_per_label list must have length {len(self.hallucination_labels)}"
                )
            for idx in range(len(hallucination_per_label)):
                if (
                    not isinstance(hallucination_per_label[idx], int)
                    or hallucination_per_label[idx] < 0
                ):
                    raise ValueError(
                        f"hallucination_per_label {hallucination_per_label[idx]} must be a nonnegative int"
                    )
            self.hallucination_per_label = hallucination_per_label
        elif isinstance(hallucination_per_label, int):
            if hallucination_per_label < 0:
                raise ValueError(
                    f"hallucination_per_label {hallucination_per_label} must be a nonnegative int"
                )
            self.hallucination_per_label = [hallucination_per_label] * len(
                self.hallucination_labels
            )
        else:
            raise ValueError(
                f"hallucination_per_label {hallucination_per_label} must be a nonnegative int or a list of nonnegative int"
            )

    def generate(self, y, y_patch_metadata):
        from collections import defaultdict

        labels = self.hallucination_labels
        slopes = self.hallucination_slopes
        intercepts = self.hallucination_intercepts
        min_widths = self.hallucination_min_widths
        max_widths = self.hallucination_max_widths
        width_means = self.hallucination_width_means
        width_stds = self.hallucination_width_stds
        num_hallucinations_per_class = self.hallucination_per_label

        y_out = []
        for i in range(len(y)):
            gs_coords = y_patch_metadata[i]["gs_coords"]
            x_min = min(gs_coords[:, 0])
            x_max = max(gs_coords[:, 0])
            y_min = min(gs_coords[:, 1])
            y_max = max(gs_coords[:, 1])

            targeted_y = defaultdict(list)
            for idx in range(len(labels)):
                targeted_y["labels"].extend(
                    num_hallucinations_per_class[idx] * [labels[idx]]
                )
                targeted_y["scores"].extend(num_hallucinations_per_class[idx] * [1.0])
                ws = np.minimum(
                    max_widths[idx],
                    np.maximum(
                        min_widths[idx],
                        width_means[idx]
                        + width_stds[idx]
                        * np.random.randn(num_hallucinations_per_class[idx]),
                    ),
                )
                hs = intercepts[idx] + slopes[idx] * ws
                lefts = np.random.uniform(
                    x_min, x_max, num_hallucinations_per_class[idx]
                )
                tops = np.random.uniform(
                    y_min, y_max, num_hallucinations_per_class[idx]
                )

                for lt, tp, w, h in zip(lefts, tops, ws, hs):
                    x0 = int(lt)
                    y0 = int(tp)
                    x1 = int(min(self.X_MAX - 1, lt + w))
                    y1 = int(min(self.Y_MAX - 1, tp + h))
                    targeted_y["boxes"].extend([[x0, y0, x1, y1]])

            targeted_y["labels"] = np.array(targeted_y["labels"])
            targeted_y["scores"] = np.array(targeted_y["scores"])
            targeted_y["boxes"] = np.array(targeted_y["boxes"])
            y_out.append(targeted_y)

        return y_out


class MatchedTranscriptLengthTargeter:
    """
    Targets labels of a length close to the true label

    If two labels are tied in length, then it pseudorandomly picks one.
    """

    def __init__(self, *, transcripts, import_from=False):
        if import_from:
            transcripts = import_from_module(import_from, transcripts)
        if not transcripts:
            raise ValueError('"transcripts" cannot be None or an empty list')
        for t in transcripts:
            if type(t) not in (bytes, str):
                raise ValueError(f"transcript type {type(t)} not in (bytes, str)")
        self.transcripts = transcripts
        self.count = 0

    def _generate(self, y):
        distances = [
            (np.abs(len(y) - len(t)), i) for (i, t) in enumerate(self.transcripts)
        ]
        distances.sort()
        min_dist, i = distances[0]
        pool = [i]
        for dist, i in distances[1:]:
            if dist == min_dist:
                pool.append(i)

        chosen_index = pool[self.count % len(pool)]
        y_target = self.transcripts[chosen_index]
        self.count += 1

        return y_target

    def generate(self, y):
        y_target = [self._generate(y_i) for y_i in y]
        if type(y) != list:  # noqa
            y_target = np.array(y_target)
        return y_target
