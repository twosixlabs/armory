import tensorflow as tf


class ImageContext:
    def __init__(self, x_shape):
        self.x_shape = x_shape
        self.input_type = tf.uint8
        self.input_min = 0
        self.input_max = 255

        self.quantization = 255

        self.output_type = tf.float32
        self.output_min = 0.0
        self.output_max = 1.0


class VideoContext(ImageContext):
    def __init__(self, x_shape, frame_rate):
        super().__init__(x_shape)
        self.frame_rate = frame_rate


carla_video_tracking_dev_context = VideoContext(
    x_shape=(None, 960, 1280, 3), frame_rate=10
)

contexts = {
    "carla_video_tracking_dev": carla_video_tracking_dev_context,
}
