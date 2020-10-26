from art.defences.preprocessor import VideoCompression
import numpy as np


class VideoCompressionNormalized(VideoCompression):
    """
    Unnormalize inputs that were normalized during preprocessing,
    process use ART VideoCompression, and renormalize
    """

    def __init__(
        self,
        video_format,
        clip_values=None,
        constant_rate_factor=28,
        channels_first=False,
        apply_fit=False,
        apply_predict=True,
        transpose=None,
        means=None,
        stds=None,
        dtype=np.float32,
        same_video=True,
    ):
        super().__init__(
            video_format=video_format,
            constant_rate_factor=constant_rate_factor,
            channels_first=channels_first,
            apply_fit=apply_fit,
            apply_predict=apply_predict,
        )
        if clip_values is None:
            clip_values = [0.0, 255.0]
        self.clip_values = clip_values

        if transpose is None:
            transpose = (0, 1, 2, 3, 4)  # identify operation
        if len(transpose) != 5:
            raise ValueError("transpose must have 5 dimensions specified")
        self.transpose = transpose

        if means is None:
            means = (0.0, 0.0, 0.0)  # identity operation
        if len(means) != 3:
            raise ValueError("means must have 3 values, one per channel")
        self.means = means

        if stds is None:
            stds = (1.0, 1.0, 1.0)  # identity operation
        if len(stds) != 3:
            raise ValueError("stds must have 3 values, one per channel")
        self.stds = stds

        self.dtype = dtype

        self.same_video = same_video

    def __call__(self, x, y=None):
        # transpose to shape NFHWC
        x = x.transpose(self.transpose)
        x = x * self.stds + self.means  # use broadcasting
        np.clip(x, self.clip_values[0], self.clip_values[1], x)
        # if batches are from same video, then stack them to reduce
        # number of video compression calls
        if self.same_video:
            x_shape = x.shape
            x = x.reshape((x_shape[0] * x_shape[1], *x_shape[2:]), order="C")
            x = np.expand_dims(x, axis=0)
        x, _ = super().__call__(x)
        if self.same_video:
            x = x.reshape(x_shape, order="C")
        x = (x - self.means) / self.stds
        x = x.astype(self.dtype)
        x = x.transpose(np.argsort(self.transpose))
        return x, y
