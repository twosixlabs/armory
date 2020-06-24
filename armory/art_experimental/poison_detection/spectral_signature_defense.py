import logging

logger = logging.getLogger(__name__)

import numpy as np

try:
    from art.defences.detector.poison import (
        GroundTruthEvaluator,
        PoisonFilteringDefence,
    )
except ImportError:
    logger.warning(
        "ART 1.2 support is deprecated and will be removed in ARMORY 0.11. Use ART 1.3"
    )
    from art.poison_detection.ground_truth_evaluator import GroundTruthEvaluator
    from art.poison_detection.poison_filtering_defence import PoisonFilteringDefence


class SpectralSignatureDefense(PoisonFilteringDefence):
    """
    Method from Tran et al., 2018 performing poisoning detection based on Spectral Signatures
    """

    defence_params = PoisonFilteringDefence.defence_params + [
        "classifier",
        "x_train",
        "y_train",
        "batch_size",
        "eps_multiplier",
        "ub_pct_poison",
    ]

    def __init__(self, classifier, x_train, y_train, **kwargs):
        """
        Create an :class:`.ActivationDefence` object with the provided classifier.
        :param classifier: Model evaluated for poison.
        :param x_train: dataset used to train the classifier.
        :param y_train: labels used to train the classifier.
        """
        super(SpectralSignatureDefense, self).__init__(classifier, x_train, y_train)
        self.set_params(**kwargs)
        self.evaluator = GroundTruthEvaluator()

    def evaluate_defence(self, is_clean, **kwargs):
        """
        If ground truth is known, this function returns a confusion matrix in the form of a JSON object.
        :param is_clean: Ground truth, where is_clean[i]=1 means that x_train[i] is clean and is_clean[i]=0 means
                         x_train[i] is poisonous.
        :param kwargs: A dictionary of defence-specific parameters.
        :return: JSON object with confusion matrix.
        """

        n_classes = self.classifier.nb_classes()
        if is_clean is None or is_clean.size == 0:
            raise ValueError(
                "is_clean was not provided while invoking evaluate_defence."
            )
        is_clean_by_class = SpectralSignatureDefense.split_by_class(
            is_clean, self.y_train, n_classes
        )
        _, predicted_clean = self.detect_poison()
        predicted_clean_by_class = SpectralSignatureDefense.split_by_class(
            predicted_clean, self.y_train, n_classes
        )

        _, conf_matrix_json = self.evaluator.analyze_correctness(
            predicted_clean_by_class, is_clean_by_class
        )

        return conf_matrix_json

    def detect_poison(self, **kwargs):
        """
        Returns poison detected and a report.
        :return: (report, is_clean_lst):
                where a report is None (for future ART compatibility)
                where is_clean is a list, where is_clean_lst[i]=1 means that x_train[i]
                there is clean and is_clean_lst[i]=0, means that x_train[i] was classified as poison.
        """

        self.set_params(**kwargs)

        n_classes = self.classifier.nb_classes()
        nb_layers = len(self.classifier.layer_names)

        features_x_poisoned = self.classifier.get_activations(
            self.x_train, layer=nb_layers - 1, batch_size=self.batch_size
        )

        features_split = SpectralSignatureDefense.split_by_class(
            features_x_poisoned, self.y_train, n_classes
        )
        keep_by_class = []
        for idx, feature in enumerate(features_split):
            score = SpectralSignatureDefense.spectral_signature_scores(feature)
            score_cutoff = np.quantile(
                score, max(1 - self.eps_multiplier * self.ub_pct_poison, 0.0)
            )
            keep_by_class.append(score < score_cutoff)

        base_indices_by_class = SpectralSignatureDefense.split_by_class(
            np.arange(self.y_train.shape[0]), self.y_train, 10
        )
        is_clean_lst = np.zeros_like(self.y_train, dtype=np.int)

        for keep_booleans, indices in zip(keep_by_class, base_indices_by_class):
            for keep_boolean, idx in zip(keep_booleans, indices):
                if keep_boolean:
                    is_clean_lst[idx] = 1

        return None, is_clean_lst

    @staticmethod
    def spectral_signature_scores(R):
        """
        :param R: Matrix of feature representations
        :return: Outlier scores for each observation based on spectral signature
        """
        M = R - np.mean(R, axis=0)
        # Following Algorithm #1, use SVD of centered features, not of covariance
        _, _, v = np.linalg.svd(M, full_matrices=False)
        eigs = v[:1]
        score = np.matmul(M, np.transpose(eigs)) ** 2
        return score

    @staticmethod
    def split_by_class(data, labels, num_classes):
        """
        :param data: Iterable of features
        :param labels: Labels, not in one-hot representations
        :param num_classes: Number of classes of labels
        :return: List of numpy arrays of features split by labels
        """
        split = [[] for _ in range(num_classes)]
        for idx, label in enumerate(labels):
            split[int(label)].append(data[idx])
        return [np.asarray(dat) for dat in split]

    def set_params(self, **kwargs):
        """
        Take in a dictionary of parameters and applies defense-specific checks before saving them as attributes.
        If a parameter is not provided, it takes its default value.
        """
        # Save defence-specific parameters
        super(SpectralSignatureDefense, self).set_params(**kwargs)

        return True
