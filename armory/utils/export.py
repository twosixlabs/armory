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
    def __init__(self, base_output_dir, num_samples):
        self.base_output_dir = base_output_dir
        self.num_samples = num_samples
        self.saved_samples = 0
        self.output_dir = None
        self.y_dict = {}

        self._make_output_dir()

    def export(
        self, x, x_adv=None, y=None, y_pred_adv=None, y_pred_clean=None, **kwargs
    ):

        if self.saved_samples < self.num_samples:

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

    def _export_so2sat(self, x, x_adv):
        for x_i, x_adv_i in zip(x, x_adv):

            if self.saved_samples == self.num_samples:
                break

            assert np.all(
                x_i.shape == x_adv_i.shape
            ), f"Benign and adversarial images are different shapes: {x_i.shape} vs. {x_adv_i.shape}"
            if x_i[..., :4].min() < -1.0 or x_i[..., :4].max() > 1.0:
                logger.warning(
                    "Benign SAR images out of expected range. Clipping to [-1, 1]."
                )
            if x_adv_i[..., :4].min() < -1.0 or x_adv_i[..., :4].max() > 1.0:
                logger.warning(
                    "Adversarial SAR images out of expected range. Clipping to [-1, 1]."
                )
            if x_i[..., 4:].min() < 0.0 or x_i[..., 4:].max() > 1.0:
                logger.warning(
                    "Benign EO images out of expected range. Clipping to [0, 1]."
                )
            if x_adv_i[..., 4:].min() < 0.0 or x_adv_i[..., 4:].max() > 1.0:
                logger.warning(
                    "Adversarial EO images out of expected range. Clipping to [0, 1]."
                )

            folder = str(self.saved_samples)
            os.mkdir(os.path.join(self.output_dir, folder))

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
            x_vv = np.log10(
                np.abs(
                    np.complex128(
                        np.clip(x_i[..., 2], -1.0, 1.0)
                        + 1j * np.clip(x_i[..., 3], -1.0, 1.0)
                    )
                    + sar_eps
                )
            )
            x_adv_vh = np.log10(
                np.abs(
                    np.complex128(
                        np.clip(x_adv_i[..., 0], -1.0, 1.0)
                        + 1j * np.clip(x_adv_i[..., 1], -1.0, 1.0)
                    )
                    + sar_eps
                )
            )
            x_adv_vv = np.log10(
                np.abs(
                    np.complex128(
                        np.clip(x_adv_i[..., 2], -1.0, 1.0)
                        + 1j * np.clip(x_adv_i[..., 3], -1.0, 1.0)
                    )
                    + sar_eps
                )
            )
            sar_min = np.min((x_vh.min(), x_vv.min(), x_adv_vh.min(), x_adv_vv.min()))
            sar_max = np.max((x_vh.max(), x_vv.max(), x_adv_vh.max(), x_adv_vv.max()))
            sar_scale = 255.0 / (sar_max - sar_min)

            benign_vh = Image.fromarray(np.uint8(sar_scale * (x_vh - sar_min)), "L")
            benign_vv = Image.fromarray(np.uint8(sar_scale * (x_vv - sar_min)), "L")
            adversarial_vh = Image.fromarray(
                np.uint8(sar_scale * (x_adv_vh - sar_min)), "L"
            )
            adversarial_vv = Image.fromarray(
                np.uint8(sar_scale * (x_adv_vv - sar_min)), "L"
            )
            benign_vh.save(os.path.join(self.output_dir, folder, "vh_benign.png"))
            benign_vv.save(os.path.join(self.output_dir, folder, "vv_benign.png"))
            adversarial_vh.save(
                os.path.join(self.output_dir, folder, "vh_adversarial.png")
            )
            adversarial_vv.save(
                os.path.join(self.output_dir, folder, "vv_adversarial.png")
            )

            eo_min = np.min((x_i[..., 4:].min(), x_adv[..., 4:].min()))
            eo_max = np.max((x_i[..., 4:].max(), x_adv[..., 4:].max()))
            eo_scale = 255.0 / (eo_max - eo_min)
            for c in range(4, 14):
                benign_eo = Image.fromarray(
                    np.uint8(eo_scale * (np.clip(x_i[..., c], 0.0, 1.0) - eo_min)), "L"
                )
                adversarial_eo = Image.fromarray(
                    np.uint8(eo_scale * (np.clip(x_adv_i[..., c], 0.0, 1.0) - eo_min)),
                    "L",
                )
                benign_eo.save(
                    os.path.join(self.output_dir, folder, f"eo{c-4}_benign.png")
                )
                adversarial_eo.save(
                    os.path.join(self.output_dir, folder, f"eo{c-4}_adversarial.png")
                )

            self.saved_samples += 1

    def _export_audio(self, x, x_adv):
        for x_i, x_adv_i in zip(x, x_adv):

            if self.saved_samples == self.num_samples:
                break

            assert np.all(
                x_i.shape == x_adv_i.shape
            ), f"Benign and adversarial audio are different shapes: {x_i.shape} vs. {x_adv_i.shape}"
            if x_i.min() < -1.0 or x_i.max() > 1.0:
                logger.warning(
                    "Benign audio out of expected range. Clipping to [-1, 1]"
                )
            if x_adv_i.min() < -1.0 or x_adv_i.max() > 1.0:
                logger.warning(
                    "Adversarial audio out of expected range. Clipping to [-1, 1]"
                )

            wavfile.write(
                os.path.join(self.output_dir, f"{self.saved_samples}_benign.wav"),
                rate=self.context.sample_rate,
                data=np.clip(x_i, -1.0, 1.0),
            )
            wavfile.write(
                os.path.join(self.output_dir, f"{self.saved_samples}_adversarial.wav"),
                rate=self.context.sample_rate,
                data=np.clip(x_adv_i, -1.0, 1.0),
            )

            self.saved_samples += 1

    def _export_video(
        self, x, x_adv, y=None, y_pred_adv=None, y_pred_clean=None, classes_to_skip=None
    ):
        for x_i, x_adv_i in zip(x, x_adv):

            plot_boxes = True if y is not None else False
            if self.saved_samples == self.num_samples:
                break

            assert np.all(
                x_i.shape == x_adv_i.shape
            ), f"Benign and adversarial videos are different shapes: {x_i.shape} vs. {x_adv_i.shape}"
            if x_i.min() < 0.0 or x_i.max() > 1.0:
                logger.warning("Benign video out of expected range. Clipping to [0, 1]")
            if x_adv_i.min() < 0.0 or x_adv_i.max() > 1.0:
                logger.warning(
                    "Adversarial video out of expected range. Clipping to [0, 1]"
                )

            folder = str(self.saved_samples)
            os.mkdir(os.path.join(self.output_dir, folder))

            benign_process = (
                ffmpeg.input(
                    "pipe:",
                    format="rawvideo",
                    pix_fmt="rgb24",
                    s=f"{x_i.shape[2]}x{x_i.shape[1]}",
                )
                .output(
                    os.path.join(self.output_dir, folder, "video_benign.mp4"),
                    pix_fmt="yuv420p",
                    vcodec="libx264",
                    r=self.context.frame_rate,
                )
                .overwrite_output()
                .run_async(pipe_stdin=True, quiet=True)
            )

            adversarial_process = (
                ffmpeg.input(
                    "pipe:",
                    format="rawvideo",
                    pix_fmt="rgb24",
                    s=f"{x_i.shape[2]}x{x_i.shape[1]}",
                )
                .output(
                    os.path.join(self.output_dir, folder, "video_adversarial.mp4"),
                    pix_fmt="yuv420p",
                    vcodec="libx264",
                    r=self.context.frame_rate,
                )
                .overwrite_output()
                .run_async(pipe_stdin=True, quiet=True)
            )

            for n_frame, (x_frame, x_adv_frame) in enumerate(zip(x_i, x_adv_i)):

                benign_pixels = np.uint8(np.clip(x_frame, 0.0, 1.0) * 255.0)
                adversarial_pixels = np.uint8(np.clip(x_adv_frame, 0.0, 1.0) * 255.0)

                benign_image = Image.fromarray(benign_pixels, "RGB")
                adversarial_image = Image.fromarray(adversarial_pixels, "RGB")
                benign_image.save(
                    os.path.join(
                        self.output_dir, folder, f"frame_{n_frame:04d}_benign.png"
                    )
                )
                adversarial_image.save(
                    os.path.join(
                        self.output_dir, folder, f"frame_{n_frame:04d}_adversarial.png"
                    )
                )

                if plot_boxes:
                    # Export images again but with ground-truth and predicted bounding boxes
                    benign_image_with_boxes = Image.fromarray(
                        np.uint8(np.clip(x_frame, 0.0, 1.0) * 255.0), "RGB"
                    )
                    adversarial_image_with_boxes = Image.fromarray(
                        np.uint8(np.clip(x_adv_frame, 0.0, 1.0) * 255.0), "RGB"
                    )

                    benign_box_layer = ImageDraw.Draw(benign_image_with_boxes)
                    adv_box_layer = ImageDraw.Draw(adversarial_image_with_boxes)

                    bbox_true = y[0]["boxes"][n_frame].astype("float32")
                    bbox_pred_adv = y_pred_adv[0]["boxes"][n_frame]

                    benign_box_layer.rectangle(bbox_true, outline="red", width=2)
                    adv_box_layer.rectangle(bbox_true, outline="red", width=2)
                    adv_box_layer.rectangle(bbox_pred_adv, outline="white", width=2)
                    if y_pred_clean is not None:
                        bbox_pred_clean = y_pred_clean[0]["boxes"][n_frame]
                        benign_box_layer.rectangle(
                            bbox_pred_clean, outline="white", width=2
                        )

                    benign_image_with_boxes.save(
                        os.path.join(
                            self.output_dir,
                            folder,
                            f"frame_{n_frame:04d}_benign_with_boxes.png",
                        )
                    )
                    adversarial_image_with_boxes.save(
                        os.path.join(
                            self.output_dir,
                            folder,
                            f"frame_{n_frame:04d}_adversarial_with_boxes.png",
                        )
                    )
                    benign_process.stdin.write(benign_image_with_boxes.tobytes())
                    adversarial_process.stdin.write(
                        adversarial_image_with_boxes.tobytes()
                    )
                else:
                    benign_process.stdin.write(benign_pixels.tobytes())
                    adversarial_process.stdin.write(adversarial_pixels.tobytes())

            benign_process.stdin.close()
            benign_process.wait()
            adversarial_process.stdin.close()
            adversarial_process.wait()
            self.saved_samples += 1


class ImageClassificationExporter(SampleExporter):
    def _export(self, x, x_adv=None, y=None, y_pred_adv=None, y_pred_clean=None):
        for i, x_i in enumerate(x):
            if self.saved_samples == self.num_samples:
                break

            self._export_image(x_i, type="benign")

            # Export adversarial image x_adv_i if present
            if x_adv is not None:
                x_adv_i = x_adv[i]
                self._export_image(x_adv_i, type="adversarial")

            self.saved_samples += 1

    def _export_image(self, x_i, type="benign"):
        if type not in ["benign", "adversarial"]:
            raise ValueError(
                f"type must be one of ['benign', 'adversarial'], received '{type}'."
            )

        if x_i.min() < 0.0 or x_i.max() > 1.0:
            logger.warning("Image out of expected range. Clipping to [0, 1].")

        # Export benign image x_i
        if x_i.shape[-1] == 1:
            self.mode = "L"
            self.x_i_mode = np.squeeze(x_i, axis=2)
        elif x_i.shape[-1] == 3:
            self.mode = "RGB"
            self.x_i_mode = x_i
        elif x_i.shape[-1] == 6:
            self.mode = "RGB"
            self.x_i_mode = x_i[..., :3]
            self.x_i_depth = x_i[..., 3:]
            depth_image = Image.fromarray(
                np.uint8(np.clip(self.x_i_depth, 0.0, 1.0) * 255.0), self.mode
            )
            depth_image.save(
                os.path.join(self.output_dir, f"{self.saved_samples}_depth_{type}.png")
            )
        else:
            raise ValueError(f"Expected 1, 3, or 6 channels, found {x_i.shape[-1]}")

        self.image = Image.fromarray(
            np.uint8(np.clip(self.x_i_mode, 0.0, 1.0) * 255.0), self.mode
        )
        self.image.save(
            os.path.join(self.output_dir, f"{self.saved_samples}_{type}.png")
        )


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
            if self.saved_samples == self.num_samples:
                break

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
        box_layer = ImageDraw.Draw(self.image)

        bboxes_true = y_i["boxes"]
        labels_true = y_i["labels"]

        bboxes_pred = y_i_pred["boxes"][y_i_pred["scores"] > score_threshold]

        for true_box, label in zip(bboxes_true, labels_true):
            if classes_to_skip is not None and label in classes_to_skip:
                continue
            box_layer.rectangle(true_box, outline="red", width=2)
        for pred_box in bboxes_pred:
            box_layer.rectangle(pred_box, outline="white", width=2)

        image.save(
            os.path.join(self.output_dir, f"{self.saved_samples}_{type}_with_boxes.png")
        )
