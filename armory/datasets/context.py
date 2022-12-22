import tensorflow as tf


class VideoContext:
    def __init__(self, x_shape, frame_rate):
        self.x_shape = x_shape
        self.input_type = tf.uint8
        self.frame_rate = frame_rate


carla_video_tracking_context = VideoContext(x_shape=(None, 960, 1280, 3), frame_rate=10)

contexts = {
    "carla_video_tracking_dev": carla_video_tracking_context,
    "carla_video_tracking_test": carla_video_tracking_context,
}
