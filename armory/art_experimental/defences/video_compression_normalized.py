from art.defences.preprocessor import VideoCompression, VideoCompressionPyTorch
import numpy as np


class VideoCompressionNormalized(VideoCompression):
    """
    Convert x from [0,1] to [0, 255] and back, if necessary
    """

    def __init__(
        self,
        video_format,
        constant_rate_factor=28,
        channels_first=False,
        apply_fit=False,
        apply_predict=True,
        verbose=False,
        dtype=np.float32,
    ):
        super().__init__(
            video_format=video_format,
            constant_rate_factor=constant_rate_factor,
            channels_first=channels_first,
            apply_fit=apply_fit,
            apply_predict=apply_predict,
            verbose=verbose,
        )
        self.dtype = dtype

    def __call__(self, x, y=None):
        scale = 1
        if x.min() >= 0 and x.max() <= 1.0:
            scale = 255

        x2 = x.copy()
        x2 *= scale
        x, _ = super().__call__(x2)
        x /= scale
        x = x.astype(self.dtype)
        return x, y


class VideoCompressionNormalizedPyTorch(VideoCompressionPyTorch):
    """
    Convert x from [0,1] to [0, 255] and back, if necessary
    """

    def __init__(
        self,
        video_format,
        constant_rate_factor=28,
        channels_first=False,
        apply_fit=False,
        apply_predict=True,
        verbose=False,
        dtype=np.float32,
    ):
        super().__init__(
            video_format=video_format,
            constant_rate_factor=constant_rate_factor,
            channels_first=channels_first,
            apply_fit=apply_fit,
            apply_predict=apply_predict,
            verbose=verbose,
        )
        self.dtype = dtype

    def forward(self, x, y=None):
        """
        Apply video compression to sample `x`.

        :param x: Sample to compress of shape NCFHW or NFHWC. `x` values are expected to be in the data range [0, 255].
        :param y: Labels of the sample `x`. This function does not affect them in any way.
        :return: Compressed sample.
        """
        scale = 1
        if x.min() >= 0 and x.max() <= 1.0:
            scale = 255

        x = x * scale
        x_compressed = self._compression_pytorch_numpy.apply(x)
        x_compressed = x_compressed / scale
        return x_compressed, y
