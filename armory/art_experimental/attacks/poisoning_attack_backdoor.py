"""
Simple data poisoning backdoor attack
"""

import numpy as np
from art.attacks import PoisoningAttack


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

    def generate(self, x, y=None, **kwargs):
        """
        :param x: Clean features of training data
        :param y: Clean labels of training data
        :param kwargs:
        :return: Generate poisoning examples and return them as an array along with information about which points are poisons
        """
        return PoisoningAttackBackdoor._generate_poison(
            x,
            y,
            self.pct_poison,
            sources=[self.source_class],
            targets=[self.target_class],
        )

    def generate_target_test(self, x_clean):
        """
        Creates test-time backdoored images by adding a square each image to induce a backdoored model
        to make targeted misclassifications
        :param x_clean: The clean images with no backdoor
        :return: The modified images with the backdoor inserted
        """
        x_backdoored = np.copy(x_clean)
        return PoisoningAttackBackdoor._add_pattern_bd(x=x_backdoored)

    def __init__(
        self, classifier, x_train, y_train, pct_poison, **kwargs,
    ):
        """
        :param classifier: ART model
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
        self.set_params(**kwargs)

    @staticmethod
    def _generate_poison(x_clean, y_clean, percent_poison, sources, targets):
        """
        Creates a backdoor in images by adding a square to the image and changing the label to a targeted
        class.
        :param x_clean: Original raw data
        :param y_clean: Original labels
        :param percent_poison: After poisoning, the target class should contain this percentage of poison
        :param sources: Array that holds the source classes for each backdoor. Poison is
        generating by taking images from the source class, adding the backdoor trigger, and labeling as the target class.
        Poisonous images from sources[i] will be labeled as targets[i].
        :param targets: This array holds the target classes for each backdoor. Poisonous images from sources[i] will be
                        labeled as targets[i].
        :return: Returns is_poison, which is a boolean array indicating which points are poisonous, x_poison, which
        contains all of the data both legitimate and poisoned, and y_poison, which contains all of the labels
        both legitimate and poisoned.
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
            num_poison = int(num_poison)
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
    def _add_pattern_bd(x, size=5, pixel_value=1, n_color_channel=3):
        """
        :param x: N X W X H X N_COLOR_CHANNEL matrix
        :param size: Size of square side in pixels
        :param pixel_value: Intensity value used for coloring square
        :param n_color_channel: Number of color channels
        :return: Matrix with square added to lower RH corner
        """
        x = np.array(x)
        shape = x.shape
        if size > shape[1] or size > shape[2]:
            raise ValueError(
                "Backdoor pattern size must be less than or equal to width and height of image"
            )

        if len(shape) == 4:
            width, height = x.shape[1:3]
            for i in range(n_color_channel):
                for j in range(size):
                    for k in range(size):
                        x[:, width - 1 - j, height - 1 - k, i] = pixel_value
        else:
            raise RuntimeError("Do not support numpy arrays of shape " + str(shape))
        return x
