"""
Simple data poisoning backdoor attack
"""

import numpy as np
from art.attacks.attack import PoisoningAttack


class PoisoningAttackBackdoor(PoisoningAttack):
    """
    Implementation of backdoor poisoning attack
    """

    attack_params = PoisoningAttack.attack_params + [
        "classifier",
        "x_train",
        "y_train",
        "pct_poison",
        "source_class",
        "target_class",
    ]

    def __init__(
        self,
        classifier,
        x_train,
        y_train,
        pct_poison,
        source_class,
        target_class,
        **kwargs,
    ):
        """
        :param classifier: Keras model
        :param x_train: Training data used for classification
        :param y_train: Training labels used for classification
        :param pct_poison: Number of poisons
        :param source_class: Class to be manipulated using backdoor trigger
        :param target_class: Goal class label for classification of source class with trigger
        :param kwargs: Extra optional keyword arguments
        :return: None
        """
        super(PoisoningAttackBackdoor, self).__init__(classifier)
        self.x_train = x_train
        self.y_train = y_train
        self.pct_poison = pct_poison
        self.source_class = source_class
        self.target_class = target_class
        self.set_params(**kwargs)

    def generate(self, x, y=None, **kwargs):
        return PoisoningAttackBackdoor._generate_poison(
            x,
            y,
            self.pct_poison,
            sources=[self.source_class],
            targets=[self.target_class],
        )

    def generate_target_test(self, x_clean):
        x_backdoored = np.copy(x_clean)

        x_backdoored = PoisoningAttackBackdoor._add_pattern_bd(x=x_backdoored)

        return x_backdoored

    @staticmethod
    def _generate_poison(x_clean, y_clean, percent_poison, sources, targets):
        """
        Creates a backdoor in MNIST images by adding a pattern or pixel to the image and changing the label to a targeted
        class.
        :param x_clean: Original raw data
        :type x_clean: `np.ndarray`
        :param y_clean: Original labels
        :type y_clean:`np.ndarray`
        :param percent_poison: After poisoning, the target class should contain this percentage of poison
        :type percent_poison: `float`
        :param sources: Array that holds the source classes for each backdoor. Poison is
        generating by taking images from the source class, adding the backdoor trigger, and labeling as the target class.
        Poisonous images from sources[i] will be labeled as targets[i].
        :type sources: `np.ndarray`
        :param targets: This array holds the target classes for each backdoor. Poisonous images from sources[i] will be
                        labeled as targets[i].
        :type targets: `np.ndarray`
        :return: Returns is_poison, which is a boolean array indicating which points are poisonous, x_poison, which
        contains all of the data both legitimate and poisoned, and y_poison, which contains all of the labels
        both legitimate and poisoned.
        :rtype: `tuple`
        """
        x_poison = np.copy(x_clean)
        y_poison = np.copy(y_clean)
        is_poison = np.zeros(np.shape(y_poison))

        for i, (src, tgt) in enumerate(zip(sources, targets)):
            n_points_in_tgt = np.size(np.where(y_clean == tgt))
            num_poison = round(
                (percent_poison * n_points_in_tgt) / (1 - percent_poison)
            )
            src_imgs = x_clean[y_clean == src]

            n_points_in_src = np.shape(src_imgs)[0]
            num_poison = int(num_poison)  # DEBUG
            indices_to_be_poisoned = np.random.choice(n_points_in_src, int(num_poison))

            imgs_to_be_poisoned = np.copy(src_imgs[indices_to_be_poisoned])
            imgs_to_be_poisoned = PoisoningAttackBackdoor._add_pattern_bd(
                x=imgs_to_be_poisoned
            )
            x_poison = np.append(x_poison, imgs_to_be_poisoned, axis=0)
            y_poison = np.append(y_poison, np.ones(num_poison) * tgt, axis=0)
            is_poison = np.append(is_poison, np.ones(num_poison))

        is_poison = is_poison != 0

        return is_poison, x_poison, y_poison

    @staticmethod
    def _add_pattern_bd(x, distance=4, pixel_value=1):
        """
        Augments a matrix by setting a checkboard-like pattern of values some `distance` away from the bottom-right
        edge to 1. Works for single images or a batch of images.
        :param x: N X W X H matrix or W X H matrix. will apply to last 2
        :type x: `np.ndarray`
        :param distance: distance from bottom-right walls. defaults to 2
        :type distance: `int`
        :param pixel_value: Value used to replace the entries of the image matrix
        :type pixel_value: `int`
        :return: augmented matrix
        :rtype: np.ndarray
        """
        x = np.array(x)
        shape = x.shape
        n_color_channel = 3
        if len(shape) == 4:
            width, height = x.shape[1:3]
            for i in range(n_color_channel):
                x[:, width - distance, height - distance, i] = pixel_value
                x[:, width - distance - 1, height - distance - 1, i] = pixel_value
                x[:, width - distance, height - distance - 2, i] = pixel_value
                x[:, width - distance - 2, height - distance, i] = pixel_value
        elif len(shape) == 3:
            width, height = x.shape[:2]
            for i in range(n_color_channel):
                x[width - distance, height - distance, i] = pixel_value
                x[width - distance - 1, height - distance - 1, i] = pixel_value
                x[width - distance, height - distance - 2, i] = pixel_value
                x[width - distance - 2, height - distance, i] = pixel_value
        else:
            raise RuntimeError("Do not support numpy arrays of shape " + str(shape))
        return x
