import os
import logging
import abc
import numpy as np
import ffmpeg
import pickle
import time
from PIL import Image, ImageDraw
from scipy.io import wavfile


logger = logging.getLogger(__name__)


class SampleExporter:
    def __init__(self, base_output_dir):
        self.base_output_dir = base_output_dir
        self.saved_batches = 0
        self.saved_samples = 0
        self.output_dir = None
        self.y_dict = {}

        self._make_output_dir()

    def export(
        self, x, x_adv=None, y=None, y_pred_adv=None, y_pred_clean=None, **kwargs
    ):
        self.y_dict[self.saved_samples] = {
            "ground truth": y,
            "predicted": y_pred_adv,
        }
        self._export(
            x=x,
            x_adv=x_adv,
            y=y,
            y_pred_adv=y_pred_adv,
            y_pred_clean=y_pred_clean,
            **kwargs,
        )

    @abc.abstractmethod
    def _export(
        self, x, x_adv=None, y=None, y_pred_adv=None, y_pred_clean=None, **kwargs
    ):
        raise NotImplementedError

    def write(self):
        """ Pickle the y_dict built up during each export() call.
            Called at end of scenario.
        """

        with open(os.path.join(self.output_dir, "predictions.pkl"), "wb") as f:
            pickle.dump(self.y_dict, f)

    def _make_output_dir(self):
        assert os.path.exists(self.base_output_dir) and os.path.isdir(
            self.base_output_dir
        ), f"Directory {self.base_output_dir} does not exist"
        assert os.access(
            self.base_output_dir, os.W_OK
        ), f"Directory {self.base_output_dir} is not writable"
        self.output_dir = os.path.join(self.base_output_dir, "saved_samples")
        if os.path.exists(self.output_dir):
            logger.warning(
                f"Sample output directory {self.output_dir} already exists. Creating new directory"
            )
            self.output_dir = os.path.join(
                self.base_output_dir, f"saved_samples_{time.time()}"
            )
        os.mkdir(self.output_dir)


class ImageClassificationExporter(SampleExporter):
    def _export(self, x, x_adv=None, y=None, y_pred_adv=None, y_pred_clean=None):
        for i, x_i in enumerate(x):
            self._export_image(x_i, type="benign")

            # Export adversarial image x_adv_i if present
            if x_adv is not None:
                x_adv_i = x_adv[i]
                self._export_image(x_adv_i, type="adversarial")

            self.saved_samples += 1
        self.saved_batches += 1

    def _export_image(self, x_i, type="benign"):
        if type not in ["benign", "adversarial"]:
            raise ValueError(
                f"type must be one of ['benign', 'adversarial'], received '{type}'."
            )
        self.image = self.get_sample(x_i)
        self.image.save(
            os.path.join(self.output_dir, f"{self.saved_samples}_{type}.png")
        )
        if x_i.shape[-1] == 6:
            self.depth_image = self.get_depth_sample(x_i)
            self.depth_image.save(
                os.path.join(self.output_dir, f"{self.saved_samples}_depth_{type}.png")
            )

    @staticmethod
    def get_sample(x_i):
        if x_i.min() < 0.0 or x_i.max() > 1.0:
            logger.warning("Image out of expected range. Clipping to [0, 1].")

        # Export benign image x_i
        if x_i.shape[-1] == 1:
            mode = "L"
            x_i_mode = np.squeeze(x_i, axis=2)
        elif x_i.shape[-1] == 3:
            mode = "RGB"
            x_i_mode = x_i
        elif x_i.shape[-1] == 6:
            mode = "RGB"
            x_i_mode = x_i[..., :3]
        else:
            raise ValueError(f"Expected 1, 3, or 6 channels, found {x_i.shape[-1]}")
        image = Image.fromarray(np.uint8(np.clip(x_i_mode, 0.0, 1.0) * 255.0), mode)
        return image

    @staticmethod
    def get_depth_sample(x_i):
        if x_i.shape[-1] != 6:
            raise ValueError(f"Expected 6 channels, found {x_i.shape[-1]}")
        x_i_depth = x_i[..., 3:]
        depth_image = Image.fromarray(
            np.uint8(np.clip(x_i_depth, 0.0, 1.0) * 255.0), "RGB"
        )
        return depth_image


class ObjectDetectionExporter(ImageClassificationExporter):
    def _export(
        self,
        x,
        x_adv=None,
        y=None,
        y_pred_adv=None,
        y_pred_clean=None,
        classes_to_skip=None,
    ):
        for i, x_i in enumerate(x):
            self._export_image(x_i, type="benign")

            y_i = y[i]
            y_i_pred_clean = y_pred_clean[i]
            self._export_image_with_boxes(
                self.image, y_i, y_i_pred_clean, type="benign"
            )

            # Export adversarial image x_adv_i if present
            if x_adv is not None:
                x_adv_i = x_adv[i]
                self._export_image(x_adv_i, type="adversarial")
                y_i_pred_adv = y_pred_adv[i]
                self._export_image_with_boxes(
                    self.image, y_i, y_i_pred_adv, type="adversarial"
                )

            self.saved_samples += 1
        self.saved_batches += 1

    @staticmethod
    def get_sample_with_boxes(
        image, y_i, y_i_pred, classes_to_skip=None, score_threshold=0.5,
    ):
        box_layer = ImageDraw.Draw(image)

        bboxes_true = y_i["boxes"]
        labels_true = y_i["labels"]

        bboxes_pred = y_i_pred["boxes"][y_i_pred["scores"] > score_threshold]

        for true_box, label in zip(bboxes_true, labels_true):
            if classes_to_skip is not None and label in classes_to_skip:
                continue
            box_layer.rectangle(true_box, outline="red", width=2)
        for pred_box in bboxes_pred:
            box_layer.rectangle(pred_box, outline="white", width=2)

        return image

    def _export_image_with_boxes(
        self,
        image,
        y_i,
        y_i_pred,
        classes_to_skip=None,
        type="benign",
        score_threshold=0.5,
    ):
        if type not in ["benign", "adversarial"]:
            raise ValueError(
                f"type must be one of ['benign', 'adversarial'], received '{type}'."
            )
        self.image_with_boxes = self.get_sample_with_boxes(
            image=image,
            y_i=y_i,
            y_i_pred=y_i_pred,
            classes_to_skip=classes_to_skip,
            score_threshold=score_threshold,
        )
        self.image_with_boxes.save(
            os.path.join(self.output_dir, f"{self.saved_samples}_{type}_with_boxes.png")
        )


class VideoClassificationExporter(SampleExporter):
    def __init__(self, base_output_dir, frame_rate):
        super().__init__(base_output_dir)
        self.frame_rate = frame_rate

    @classmethod
    def from_context(cls, base_output_dir, context):
        return cls(base_output_dir, context.frame_rate)

    def _export(
        self, x, x_adv=None, y=None, y_pred_adv=None, y_pred_clean=None, **kwargs
    ):
        for i, x_i in enumerate(x):
            self._export_video(x_i, type="benign")

            if x_adv is not None:
                x_adv_i = x_adv[i]
                self._export_video(x_adv_i, type="adversarial")

            self.saved_samples += 1
        self.saved_batches += 1

    def _export_video(self, x_i, type="benign"):
        if type not in ["benign", "adversarial"]:
            raise ValueError(
                f"type must be one of ['benign', 'adversarial'], received '{type}'."
            )
        folder = str(self.saved_samples)
        os.makedirs(os.path.join(self.output_dir, folder), exist_ok=True)

        ffmpeg_process = (
            ffmpeg.input(
                "pipe:",
                format="rawvideo",
                pix_fmt="rgb24",
                s=f"{x_i.shape[2]}x{x_i.shape[1]}",
            )
            .output(
                os.path.join(self.output_dir, folder, f"video_{type}.mp4"),
                pix_fmt="yuv420p",
                vcodec="libx264",
                r=self.frame_rate,
            )
            .overwrite_output()
            .run_async(pipe_stdin=True, quiet=True)
        )

        self.frames = self.get_sample(x_i)

        for n_frame, frame in enumerate(self.frames):
            pixels = np.array(frame)
            ffmpeg_process.stdin.write(pixels.tobytes())
            frame.save(
                os.path.join(self.output_dir, folder, f"frame_{n_frame:04d}_{type}.png")
            )

        ffmpeg_process.stdin.close()
        ffmpeg_process.wait()

    @staticmethod
    def get_sample(x_i):
        if x_i.min() < 0.0 or x_i.max() > 1.0:
            logger.warning("video out of expected range. Clipping to [0, 1]")

        pil_frames = []
        for n_frame, x_frame, in enumerate(x_i):
            pixels = np.uint8(np.clip(x_frame, 0.0, 1.0) * 255.0)
            image = Image.fromarray(pixels, "RGB")
            pil_frames.append(image)
        return pil_frames


class VideoTrackingExporter(VideoClassificationExporter):
    def _export(
        self, x, x_adv=None, y=None, y_pred_adv=None, y_pred_clean=None, **kwargs
    ):
        for i, x_i in enumerate(x):
            self._export_video(x_i, type="benign")

            y_i = y[i]
            y_i_pred_clean = y_pred_clean[i]
            self._export_video_with_boxes(x_i, y_i, y_i_pred_clean, type="benign")

            if x_adv is not None:
                x_adv_i = x_adv[i]
                self._export_video(x_adv_i, type="adversarial")
                y_i_pred_adv = y_pred_adv[i]
                self._export_video_with_boxes(
                    x_adv_i, y_i, y_i_pred_adv, type="adversarial"
                )

            self.saved_samples += 1
        self.saved_batches += 1

    def _export_video_with_boxes(self, x_i, y_i, y_i_pred, type="benign"):
        if type not in ["benign", "adversarial"]:
            raise ValueError(
                f"type must be one of ['benign', 'adversarial'], received '{type}'."
            )

        folder = str(self.saved_samples)
        os.makedirs(os.path.join(self.output_dir, folder), exist_ok=True)

        ffmpeg_process = (
            ffmpeg.input(
                "pipe:",
                format="rawvideo",
                pix_fmt="rgb24",
                s=f"{x_i.shape[2]}x{x_i.shape[1]}",
            )
            .output(
                os.path.join(self.output_dir, folder, f"video_{type}_with_boxes.mp4"),
                pix_fmt="yuv420p",
                vcodec="libx264",
                r=self.frame_rate,
            )
            .overwrite_output()
            .run_async(pipe_stdin=True, quiet=True)
        )

        self.frames_with_boxes = self.get_sample_with_boxes(
            x_i=x_i, y_i=y_i, y_i_pred=y_i_pred
        )
        for n_frame, frame, in enumerate(self.frames_with_boxes):
            frame.save(
                os.path.join(
                    self.output_dir,
                    folder,
                    f"frame_{n_frame:04d}_{type}_with_boxes.png",
                )
            )
            pixels_with_boxes = np.array(frame)
            ffmpeg_process.stdin.write(pixels_with_boxes.tobytes())

        ffmpeg_process.stdin.close()
        ffmpeg_process.wait()

    @staticmethod
    def get_sample_with_boxes(x_i, y_i, y_i_pred):
        if x_i.min() < 0.0 or x_i.max() > 1.0:
            logger.warning("video out of expected range. Clipping to [0,1]")

        pil_frames = []
        for n_frame, x_frame, in enumerate(x_i):
            pixels = np.uint8(np.clip(x_frame, 0.0, 1.0) * 255.0)
            image = Image.fromarray(pixels, "RGB")
            box_layer = ImageDraw.Draw(image)
            bbox_true = y_i["boxes"][n_frame].astype("float32")
            bbox_pred = y_i_pred["boxes"][n_frame]
            box_layer.rectangle(bbox_true, outline="red", width=2)
            box_layer.rectangle(bbox_pred, outline="white", width=2)
            pil_frames.append(image)

        return pil_frames


class AudioExporter(SampleExporter):
    def __init__(self, base_output_dir, sample_rate):
        self.sample_rate = sample_rate
        super().__init__(base_output_dir)

    @classmethod
    def from_context(cls, base_output_dir, context):
        return cls(base_output_dir, context.sample_rate)

    def _export(
        self, x, x_adv=None, y=None, y_pred_adv=None, y_pred_clean=None, **kwargs
    ):
        for i, x_i in enumerate(x):
            self._export_audio(x_i, type="benign")

            if x_adv is not None:
                x_i_adv = x_adv[i]
                self._export_audio(x_i_adv, type="adversarial")

            self.saved_samples += 1
        self.saved_batches += 1

    def _export_audio(self, x_i, type="benign"):
        if type not in ["benign", "adversarial"]:
            raise ValueError(
                f"type must be one of ['benign', 'adversarial'], received '{type}'."
            )

        if x_i.min() < -1.0 or x_i.max() > 1.0:
            logger.warning("input out of expected range. Clipping to [-1, 1]")

        wavfile.write(
            os.path.join(self.output_dir, f"{self.saved_samples}_{type}.wav"),
            rate=self.sample_rate,
            data=np.clip(x_i, -1.0, 1.0),
        )


class So2SatExporter(SampleExporter):
    def _export(
        self, x, x_adv=None, y=None, y_pred_adv=None, y_pred_clean=None, **kwargs
    ):

        for i, x_i in enumerate(x):
            self._export_so2sat_image(x_i, type="benign")

            if x_adv is not None:
                x_adv_i = x_adv[i]
                self._export_so2sat_image(x_adv_i, type="adversarial")

            self.saved_samples += 1
        self.saved_batches += 1

    def _export_so2sat_image(self, x_i, type="benign"):
        if type not in ["benign", "adversarial"]:
            raise ValueError(
                f"type must be one of ['benign', 'adversarial'], received '{type}'."
            )

        folder = str(self.saved_samples)
        os.makedirs(os.path.join(self.output_dir, folder), exist_ok=True)

        self.vh_image = self.get_vh_sample(x_i)
        self.vh_image.save(os.path.join(self.output_dir, folder, f"vh_{type}.png"))

        self.vv_image = self.get_vv_sample(x_i)
        self.vv_image.save(os.path.join(self.output_dir, folder, f"vv_{type}.png"))

        self.eo_images = self.get_eo_samples(x_i)
        for i in range(10):
            eo_image = self.eo_images[i]
            eo_image.save(os.path.join(self.output_dir, folder, f"eo{i}_{type}.png"))

    @staticmethod
    def get_vh_sample(x_i):
        if x_i[..., :4].min() < -1.0 or x_i[..., :4].max() > 1.0:
            logger.warning("SAR image out of expected range. Clipping to [-1, 1].")
        if x_i[..., 4:].min() < 0.0 or x_i[..., 4:].max() > 1.0:
            logger.warning("EO image out of expected range. Clipping to [0, 1].")

        sar_eps = 1e-9 + 1j * 1e-9
        x_vh = np.log10(
            np.abs(
                np.complex128(
                    np.clip(x_i[..., 0], -1.0, 1.0)
                    + 1j * np.clip(x_i[..., 1], -1.0, 1.0)
                )
                + sar_eps
            )
        )
        sar_min = x_vh.min()
        sar_max = x_vh.max()
        sar_scale = 255.0 / (sar_max - sar_min)

        vh_image = Image.fromarray(np.uint8(sar_scale * (x_vh - sar_min)), "L")
        return vh_image

    @staticmethod
    def get_vv_sample(x_i):
        if x_i[..., :4].min() < -1.0 or x_i[..., :4].max() > 1.0:
            logger.warning("SAR image out of expected range. Clipping to [-1, 1].")
        if x_i[..., 4:].min() < 0.0 or x_i[..., 4:].max() > 1.0:
            logger.warning("EO image out of expected range. Clipping to [0, 1].")

        sar_eps = 1e-9 + 1j * 1e-9
        x_vv = np.log10(
            np.abs(
                np.complex128(
                    np.clip(x_i[..., 2], -1.0, 1.0)
                    + 1j * np.clip(x_i[..., 3], -1.0, 1.0)
                )
                + sar_eps
            )
        )
        sar_min = x_vv.min()
        sar_max = x_vv.max()
        sar_scale = 255.0 / (sar_max - sar_min)

        vv_image = Image.fromarray(np.uint8(sar_scale * (x_vv - sar_min)), "L")
        return vv_image

    @staticmethod
    def get_eo_samples(x_i):
        if x_i[..., :4].min() < -1.0 or x_i[..., :4].max() > 1.0:
            logger.warning("SAR image out of expected range. Clipping to [-1, 1].")
        if x_i[..., 4:].min() < 0.0 or x_i[..., 4:].max() > 1.0:
            logger.warning("EO image out of expected range. Clipping to [0, 1].")

        eo_images = []

        eo_min = x_i[..., 4:].min()
        eo_max = x_i[..., 4:].max()
        eo_scale = 255.0 / (eo_max - eo_min)
        for c in range(4, 14):
            eo = Image.fromarray(
                np.uint8(eo_scale * (np.clip(x_i[..., c], 0.0, 1.0) - eo_min)), "L"
            )
            eo_images.append(eo)

        return eo_images
