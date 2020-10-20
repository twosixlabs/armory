from art.defences.preprocessor import JpegCompression
import numpy as np


class JpegCompressionMultiChannelImage(JpegCompression):
    """
    Unnormalize inputs that were normalized during preprocessing,
    process use ART JpegCompression, and renormalize
    """

    def __init__(
        self,
        clip_values,
        quality=50,
        channel_index=3,
        apply_fit=True,
        apply_predict=False,
        means=None,
        stds=None,
        n_channels=14,
    ):
        super().__init__(
            clip_values,
            quality=quality,
            channel_index=channel_index,
            apply_fit=apply_fit,
            apply_predict=apply_predict,
        )
        if means is None:
            means = tuple(0.0 for _ in range(n_channels))  # identity operation
        self.means = means

        if stds is None:
            stds = tuple(1.0 for _ in range(n_channels))  # identity operation
        self.stds = stds

    def __call__(self, x, y=None):
        x = (x - self.means) / self.stds
        np.clip(x, self.clip_values[0], self.clip_values[1], x)
        x = np.transpose(x, (0, 3, 1, 2))  # Change from nhwc to nchw
        x = np.expand_dims(x, axis=-1)
        x, _ = super().__call__(x)
        x = x * self.stds + self.means
        x = np.transpose(x[..., 0], (0, 2, 3, 1))
        return x, y
