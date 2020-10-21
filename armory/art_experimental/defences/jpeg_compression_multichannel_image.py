from art.defences.preprocessor import JpegCompression
import numpy as np


class JpegCompressionMultiChannelImage(JpegCompression):
    """
    Rescale inputs that may not be in [0,1] after preprocessing,
    process use ART JpegCompression treating input as video
    (so that number of channels need not be 1,3), scale back
    to original preprocessing
    """

    def __init__(
        self,
        clip_values,
        quality=50,
        channel_index=3,
        apply_fit=True,
        apply_predict=False,
        mins=None,
        ranges=None,
        n_channels=14,
    ):
        super().__init__(
            clip_values,
            quality=quality,
            channel_index=channel_index,
            apply_fit=apply_fit,
            apply_predict=apply_predict,
        )
        if mins is None:
            mins = (0.0,) * n_channels  # identity operation
        self.mins = mins

        if ranges is None:
            ranges = (1.0,) * n_channels  # identity operation
        self.ranges = ranges

    def __call__(self, x, y=None):
        x = (x - self.mins) / self.ranges
        x = np.transpose(x, (0, 3, 1, 2))  # Change from nhwc to nchw
        x = np.expand_dims(x, axis=-1)
        x, _ = super().__call__(x)
        x = x * self.ranges + self.mins
        x = np.transpose(x[..., 0], (0, 2, 3, 1))
        return x, y
