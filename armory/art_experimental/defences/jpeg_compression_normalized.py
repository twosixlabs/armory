from art.defences.preprocessor import JpegCompression
import numpy as np


class JpegCompressionNormalized(JpegCompression):
    """
    Unnormalize inputs that were normalized during preprocessing,
    process use ART JpegCompression, and renormalize
    """

    def __init__(
        self,
        clip_values,
        quality=50,
        apply_fit=True,
        apply_predict=False,
        means=None,
        stds=None,
        dtype=np.float32,
    ):
        super().__init__(
            clip_values,
            quality=quality,
            apply_fit=apply_fit,
            apply_predict=apply_predict,
        )
        if means is None:
            means = (0.0, 0.0, 0.0)  # identity operation
        if len(means) != 3:
            raise ValueError("means must have 3 values, one per channel")
        self.means = np.array(means, dtype=dtype)

        if stds is None:
            stds = (1.0, 1.0, 1.0)  # identity operation
        if len(stds) != 3:
            raise ValueError("stds must have 3 values, one per channel")
        self.stds = np.array(stds, dtype=dtype)

    def __call__(self, x, y=None):
        x = x * self.stds + self.means
        np.clip(x, self.clip_values[0], self.clip_values[1], x)
        x, _ = super().__call__(x)
        x = (x - self.means) / self.stds
        return x, y
