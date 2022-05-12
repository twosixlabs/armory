import os
import abc
import numpy as np
import ffmpeg
import pickle
import time
from PIL import Image, ImageDraw
from scipy.io import wavfile
import json
from copy import deepcopy

from armory.logs import log
from armory.instrument import Meter


class SampleExporter:
    def __init__(self, base_output_dir, default_export_kwargs={}):
        self.base_output_dir = base_output_dir
        self.output_dir = None
        self.y_dict = {}
        self.default_export_kwargs = default_export_kwargs
        self._set_output_dir()

    def export(self, x_i, basename, **kwargs):
        export_kwargs = dict(
            list(self.default_export_kwargs.items()) + list(kwargs.items())
        )
        if not os.path.exists(self.output_dir):
            self._make_output_dir()

        self._export(
            x_i,
            basename,
            **export_kwargs,
        )

    @abc.abstractmethod
    def _export(self, x_i, prefix, **kwargs):
        raise NotImplementedError(
            f"_export() method should be defined for export class {self.__class__}"
        )

    @abc.abstractmethod
    def get_sample(self):
        raise NotImplementedError(
            f"get_sample() method should be defined for export class {self.__class__}"
        )

    def _set_output_dir(self):
        assert os.path.exists(self.base_output_dir) and os.path.isdir(
            self.base_output_dir
        ), f"Directory {self.base_output_dir} does not exist"
        assert os.access(
            self.base_output_dir, os.W_OK
        ), f"Directory {self.base_output_dir} is not writable"
        self.output_dir = os.path.join(self.base_output_dir, "saved_samples")
        if os.path.exists(self.output_dir):
            log.warning(
                f"Sample output directory {self.output_dir} already exists, will create new directory"
            )
            self.output_dir = os.path.join(
                self.base_output_dir, f"saved_samples_{time.time()}"
            )

    def _make_output_dir(self):
        os.mkdir(self.output_dir)


class ImageClassificationExporter(SampleExporter):
    def __init__(self, base_output_dir, default_export_kwargs={}):
        super().__init__(
            base_output_dir=base_output_dir, default_export_kwargs=default_export_kwargs
        )
        self.file_extension = ".png"

    def _export(self, x, basename):
        self.image = self.get_sample(x)
        self.image.save(
            os.path.join(
                self.output_dir,
                f"{basename}{self.file_extension}",
            )
        )
        if x.shape[-1] == 6:
            self.depth_image = self.get_sample(x[..., 3:])
            self.depth_image.save(
                os.path.join(
                    self.output_dir,
                    f"{basename}_depth{self.file_extension}",
                )
            )

    @staticmethod
    def get_sample(x):
        """

        :param x: floating point np array of shape (H, W, C) in [0.0, 1.0], where C = 1 (grayscale),
                3 (RGB), or 6 (RGB-Depth)
        :return: PIL.Image.Image
        """

        if x.min() < 0.0 or x.max() > 1.0:
            log.warning("Image out of expected range. Clipping to [0, 1].")

        # Export benign image x_i
        if x.shape[-1] == 1:
            mode = "L"
            x_i_mode = np.squeeze(x, axis=2)
        elif x.shape[-1] == 3:
            mode = "RGB"
            x_i_mode = x
        elif x.shape[-1] == 6:
            mode = "RGB"
            x_i_mode = x[..., :3]
        else:
            raise ValueError(f"Expected 1, 3, or 6 channels, found {x.shape[-1]}")
        image = Image.fromarray(np.uint8(np.clip(x_i_mode, 0.0, 1.0) * 255.0), mode)
        return image


class ObjectDetectionExporter(ImageClassificationExporter):
    def __init__(self, base_output_dir, default_export_kwargs={}):
        super().__init__(base_output_dir, default_export_kwargs)
        self.ground_truth_boxes_coco_format = []
        self.benign_predicted_boxes_coco_format = []
        self.adversarial_predicted_boxes_coco_format = []

    def _export(
        self,
        x,
        basename,
        with_boxes=False,
        y=None,
        y_pred=None,
        score_threshold=0.5,
        classes_to_skip=None,
    ):
        if with_boxes:
            self.image_with_boxes = self.get_sample(
                x,
                with_boxes=True,
                y=y,
                y_pred=y_pred,
                score_threshold=score_threshold,
                classes_to_skip=classes_to_skip,
            )
            fname_with_boxes = f"{basename}_with_boxes{self.file_extension}"
            self.image_with_boxes.save(os.path.join(self.output_dir, fname_with_boxes))
        else:
            super()._export(x, basename)

    # TODO: this method isn't being used anymore
    def _export_image(
        self,
        x_i,
        name="benign",
        with_boxes=False,
        y_i=None,
        y_i_pred=None,
        score_threshold=0.5,
        classes_to_skip=None,
    ):
        if not with_boxes:
            super()._export_image(x_i=x_i, name=name)
            return

        self.image_with_boxes = self.get_sample(
            x=x_i,
            with_boxes=True,
            y=y_i,
            y_pred=y_i_pred,
            classes_to_skip=classes_to_skip,
            score_threshold=score_threshold,
        )
        self.image_with_boxes.save(
            os.path.join(self.output_dir, f"{self.saved_samples}_{name}_with_boxes.png")
        )

        if y_i is not None:
            # Can only export box annotations if we have 'image_id' from y_i
            gt_boxes_coco, pred_boxes_coco = self.get_coco_formatted_bounding_box_data(
                y_i=y_i,
                y_i_pred=y_i_pred,
                classes_to_skip=classes_to_skip,
                score_threshold=score_threshold,
            )

            # Add coco box dictionaries to correct lists
            if name == "benign":
                for coco_box in pred_boxes_coco:
                    self.benign_predicted_boxes_coco_format.append(coco_box)
                for coco_box in gt_boxes_coco:
                    self.ground_truth_boxes_coco_format.append(coco_box)
            elif name == "adversarial":
                # don't save gt boxes here since they are the same as for benign
                for coco_box in pred_boxes_coco:
                    self.adversarial_predicted_boxes_coco_format.append(coco_box)

    def get_coco_formatted_bounding_box_data(
        self, y_i, y_i_pred=None, score_threshold=0.5, classes_to_skip=None
    ):
        """
        :param y_i: ground-truth label dict
        :param y_i_pred: predicted label dict
        :param score_threshold: float in [0, 1]; predicted boxes with confidence > score_threshold are exported
        :param classes_to_skip: List[Int] containing class ID's for which boxes should not be exported
        :return: Two lists of dictionaries, containing coco-formatted bounding box data for ground truth and predicted labels
        """

        ground_truth_boxes_coco_format = []
        predicted_boxes_coco_format = []

        image_id = y_i["image_id"][0]  # All boxes in y_i are for the same image
        bboxes_true = y_i["boxes"]
        labels_true = y_i["labels"]

        for true_box, label in zip(bboxes_true, labels_true):
            if classes_to_skip is not None and label in classes_to_skip:
                continue
            xmin, ymin, xmax, ymax = true_box
            ground_truth_box_coco = {
                "image_id": int(image_id),
                "category_id": int(label),
                "bbox": [int(xmin), int(ymin), int(xmax - xmin), int(ymax - ymin)],
            }
            ground_truth_boxes_coco_format.append(ground_truth_box_coco)

        if y_i_pred is not None:
            bboxes_pred = y_i_pred["boxes"][y_i_pred["scores"] > score_threshold]
            labels_pred = y_i_pred["labels"][y_i_pred["scores"] > score_threshold]
            scores_pred = y_i_pred["scores"][y_i_pred["scores"] > score_threshold]

            for pred_box, label, score in zip(bboxes_pred, labels_pred, scores_pred):
                xmin, ymin, xmax, ymax = pred_box
                predicted_box_coco = {
                    "image_id": int(image_id),
                    "category_id": int(label),
                    "bbox": [int(xmin), int(ymin), int(xmax - xmin), int(ymax - ymin)],
                    "score": float(score),
                }
                predicted_boxes_coco_format.append(predicted_box_coco)
        else:
            log.warning(
                "Annotations for predicted bounding boxes will not be exported.  Provide y_i_pred if this is not desired."
            )

        return ground_truth_boxes_coco_format, predicted_boxes_coco_format

    def get_sample(
        self,
        x,
        with_boxes=False,
        y=None,
        y_pred=None,
        score_threshold=0.5,
        classes_to_skip=None,
    ):
        """
        :param x:  floating point np array of shape (H, W, C) in [0.0, 1.0], where C = 1 (grayscale),
                3 (RGB), or 6 (RGB-Depth)
        :param with_boxes: boolean indicating whether to display bounding boxes
        :param y: ground-truth label dict
        :param y_pred: predicted label dict
        :param score_threshold: float in [0, 1]; boxes with confidence > score_threshold are displayed
        :param classes_to_skip: List[Int] containing class ID's for which boxes should not be displayed
        :return: PIL.Image.Image
        """
        image = super().get_sample(x)
        if not with_boxes:
            return image

        if y is None and y_pred is None:
            raise TypeError("Both y_i and y_i_pred are None, but with_boxes is True")
        box_layer = ImageDraw.Draw(image)

        if y is not None:
            bboxes_true = y["boxes"]
            labels_true = y["labels"]

            for true_box, label in zip(bboxes_true, labels_true):
                if classes_to_skip is not None and label in classes_to_skip:
                    continue
                box_layer.rectangle(true_box, outline="red", width=2)

        if y_pred is not None:
            bboxes_pred = y_pred["boxes"][y_pred["scores"] > score_threshold]

            for pred_box in bboxes_pred:
                box_layer.rectangle(pred_box, outline="white", width=2)

        return image

    def write(self):
        super().write()
        if len(self.ground_truth_boxes_coco_format) > 0:
            json.dump(
                self.ground_truth_boxes_coco_format,
                open(
                    os.path.join(
                        self.output_dir, "ground_truth_boxes_coco_format.json"
                    ),
                    "w",
                ),
            )
        if len(self.benign_predicted_boxes_coco_format) > 0:
            json.dump(
                self.benign_predicted_boxes_coco_format,
                open(
                    os.path.join(
                        self.output_dir, "benign_predicted_boxes_coco_format.json"
                    ),
                    "w",
                ),
            )
        if len(self.adversarial_predicted_boxes_coco_format) > 0:
            json.dump(
                self.adversarial_predicted_boxes_coco_format,
                open(
                    os.path.join(
                        self.output_dir, "adversarial_predicted_boxes_coco_format.json"
                    ),
                    "w",
                ),
            )


class DApricotExporter(ObjectDetectionExporter):
    def _export(
        self,
        x,
        x_adv=None,
        with_boxes=False,
        y=None,
        y_pred_adv=None,
        y_pred_clean=None,
        score_threshold=0.5,
        classes_to_skip=[12],  # class of green-screen
    ):
        if x_adv is None:
            raise TypeError("Expected x_adv to not be None for DApricot scenario.")
        x_adv_angle_1 = x_adv[0]
        self._export_image(x_adv_angle_1, name="adversarial_angle_1")

        x_adv_angle_2 = x_adv[1]
        self._export_image(x_adv_angle_2, name="adversarial_angle_2")

        x_adv_angle_3 = x_adv[2]
        self._export_image(x_adv_angle_3, name="adversarial_angle_3")

        if with_boxes:
            y_angle_1 = y[0]
            y_pred_angle_1 = deepcopy(y_pred_adv[0])
            y_pred_angle_1["boxes"] = self.convert_boxes_tf_to_torch(
                x_adv_angle_1, y_pred_angle_1["boxes"]
            )
            self._export_image(
                x_adv_angle_1,
                with_boxes=True,
                y_i=y_angle_1,
                y_i_pred=y_pred_angle_1,
                score_threshold=score_threshold,
                classes_to_skip=classes_to_skip,
                name="adversarial_angle_1",
            )

            y_angle_2 = y[1]
            y_pred_angle_2 = deepcopy(y_pred_adv[1])
            y_pred_angle_2["boxes"] = self.convert_boxes_tf_to_torch(
                x_adv_angle_2, y_pred_angle_2["boxes"]
            )
            self._export_image(
                x_adv_angle_2,
                with_boxes=True,
                y_i=y_angle_2,
                y_i_pred=y_pred_angle_2,
                score_threshold=score_threshold,
                classes_to_skip=classes_to_skip,
                name="adversarial_angle_2",
            )

            y_angle_3 = y[2]
            y_pred_angle_3 = deepcopy(y_pred_adv[2])
            y_pred_angle_3["boxes"] = self.convert_boxes_tf_to_torch(
                x_adv_angle_3, y_pred_angle_3["boxes"]
            )
            self._export_image(
                x_adv_angle_3,
                with_boxes=True,
                y_i=y_angle_3,
                y_i_pred=y_pred_angle_3,
                score_threshold=score_threshold,
                classes_to_skip=classes_to_skip,
                name="adversarial_angle_3",
            )

        self.saved_samples += 1
        self.saved_batches += 1

    @staticmethod
    def convert_boxes_tf_to_torch(x, box_array):
        # Convert boxes from [y1/height, x1/width, y2/height, x2/width] to [x1, y1, x2, y2]
        if box_array.max() > 1:
            log.warning(
                "Attempting to scale boxes from [0, 1] to [0, 255], but boxes are already outside [0, 1]."
            )
        converted_boxes = box_array[:, [1, 0, 3, 2]]
        height, width = x.shape[:2]
        return (converted_boxes * [width, height, width, height]).astype(np.float32)


class VideoClassificationExporter(SampleExporter):
    def __init__(self, base_output_dir, frame_rate, default_export_kwargs={}):
        super().__init__(base_output_dir, default_export_kwargs=default_export_kwargs)
        self.frame_rate = frame_rate
        self.video_file_extension = ".mp4"
        self.frame_file_extension = ".png"

    def _export(self, x, basename):
        folder = str(basename)
        os.makedirs(os.path.join(self.output_dir, folder), exist_ok=True)

        ffmpeg_process = (
            ffmpeg.input(
                "pipe:",
                format="rawvideo",
                pix_fmt="rgb24",
                s=f"{x.shape[2]}x{x.shape[1]}",
            )
            .output(
                os.path.join(
                    self.output_dir,
                    folder,
                    f"video_{basename}{self.video_file_extension}",
                ),
                pix_fmt="yuv420p",
                vcodec="libx264",
                r=self.frame_rate,
            )
            .overwrite_output()
            .run_async(pipe_stdin=True, quiet=True)
        )

        self.frames = self.get_sample(x)

        for n_frame, frame in enumerate(self.frames):
            pixels = np.array(frame)
            ffmpeg_process.stdin.write(pixels.tobytes())
            frame.save(
                os.path.join(
                    self.output_dir,
                    folder,
                    f"{basename}_frame_{n_frame:04d}{self.frame_file_extension}",
                )
            )

        ffmpeg_process.stdin.close()
        ffmpeg_process.wait()

    @staticmethod
    def get_sample(x):
        """

        :param x: floating point np array of shape (num_frames, H, W, C=3) in [0.0, 1.0]
        :return: List[PIL.Image.Image] of length equal to num_frames
        """
        if x.min() < 0.0 or x.max() > 1.0:
            log.warning("video out of expected range. Clipping to [0, 1]")

        pil_frames = []
        for n_frame, x_frame in enumerate(x):
            pixels = np.uint8(np.clip(x_frame, 0.0, 1.0) * 255.0)
            image = Image.fromarray(pixels, "RGB")
            pil_frames.append(image)
        return pil_frames


class VideoTrackingExporter(VideoClassificationExporter):
    def _export(self, x, basename, with_boxes=False, y=None, y_pred=None):
        if with_boxes:
            folder = str(basename)
            os.makedirs(os.path.join(self.output_dir, folder), exist_ok=True)

            ffmpeg_process = (
                ffmpeg.input(
                    "pipe:",
                    format="rawvideo",
                    pix_fmt="rgb24",
                    s=f"{x.shape[2]}x{x.shape[1]}",
                )
                .output(
                    os.path.join(
                        self.output_dir,
                        folder,
                        f"video_{basename}_with_boxes{self.video_file_extension}",
                    ),
                    pix_fmt="yuv420p",
                    vcodec="libx264",
                    r=self.frame_rate,
                )
                .overwrite_output()
                .run_async(pipe_stdin=True, quiet=True)
            )
            self.frames_with_boxes = self.get_sample(
                x, with_boxes=True, y=y, y_pred=y_pred
            )
            for n_frame, frame in enumerate(self.frames_with_boxes):
                frame.save(
                    os.path.join(
                        self.output_dir,
                        folder,
                        f"{basename}_frame_{n_frame:04d}_with_boxes{self.frame_file_extension}",
                    )
                )
                pixels_with_boxes = np.array(frame)
                ffmpeg_process.stdin.write(pixels_with_boxes.tobytes())

            ffmpeg_process.stdin.close()
            ffmpeg_process.wait()
        else:
            super()._export(x, basename)

    def get_sample(self, x, with_boxes=False, y=None, y_pred=None):
        """

        :param x: floating point np array of shape (num_frames, H, W, C=3) in [0.0, 1.0]
        :param with_boxes: boolean indicating whether to display bounding boxes
        :param y: ground-truth label dict
        :param y_pred: predicted label dict
        :return: List[PIL.Image.Image] of length equal to num_frames
        """
        if not with_boxes:
            return super().get_sample(x)

        if y is None and y_pred is None:
            raise TypeError("Both y_i and y_pred are None, but with_boxes is True.")
        if x.min() < 0.0 or x.max() > 1.0:
            log.warning("video out of expected range. Clipping to [0,1]")

        pil_frames = []
        for n_frame, x_frame in enumerate(x):
            pixels = np.uint8(np.clip(x_frame, 0.0, 1.0) * 255.0)
            image = Image.fromarray(pixels, "RGB")
            box_layer = ImageDraw.Draw(image)
            if y is not None:
                bbox_true = y["boxes"][n_frame].astype("float32")
                box_layer.rectangle(bbox_true, outline="red", width=2)
            if y_pred is not None:
                bbox_pred = y_pred["boxes"][n_frame]
                box_layer.rectangle(bbox_pred, outline="white", width=2)
            pil_frames.append(image)

        return pil_frames


class AudioExporter(SampleExporter):
    def __init__(self, base_output_dir, sample_rate):
        self.sample_rate = sample_rate
        self.file_extension = ".wav"
        super().__init__(base_output_dir)

    def _export(self, x, basename):
        x_copy = deepcopy(x)

        if not np.isfinite(x_copy).all():
            posinf, neginf = 1, -1
            log.warning(
                f"audio vector has infinite values. Mapping nan to 0, -inf to {neginf}, inf to {posinf}."
            )
            x_copy = np.nan_to_num(x_copy, posinf=posinf, neginf=neginf)

        if x_copy.min() < -1.0 or x_copy.max() > 1.0:
            log.warning(
                "audio vector out of expected [-1, 1] range, normalizing by the max absolute value"
            )
            x_copy = x_copy / np.abs(x_copy).max()

        wavfile.write(
            os.path.join(self.output_dir, f"{basename}{self.file_extension}"),
            rate=self.sample_rate,
            data=x_copy,
        )

    @staticmethod
    def get_sample(x, dataset_context):
        """

        :param x: floating point np array of shape (sequence_length,) in [-1.0, 1.0]
        :param dataset_context: armory.data.datasets AudioContext object
        :return: int np array of shape (sequence_length, )
        """
        assert dataset_context.input_type == np.int64
        assert dataset_context.quantization == 2**15

        return np.clip(
            np.int16(x * dataset_context.quantization),
            dataset_context.input_min,
            dataset_context.input_max,
        )


class So2SatExporter(SampleExporter):
    def __init__(self, base_output_dir, default_export_kwargs={}):
        super().__init__(
            base_output_dir=base_output_dir, default_export_kwargs=default_export_kwargs
        )
        self.file_extension = ".png"

    def _export(self, x, basename):
        folder = str(basename)
        os.makedirs(os.path.join(self.output_dir, folder), exist_ok=True)

        self.vh_image = self.get_sample(x, modality="vh")
        self.vh_image.save(
            os.path.join(self.output_dir, folder, f"{basename}_vh{self.file_extension}")
        )

        self.vv_image = self.get_sample(x, modality="vv")
        self.vv_image.save(
            os.path.join(self.output_dir, folder, f"{basename}_vv{self.file_extension}")
        )

        self.eo_images = self.get_sample(x, modality="eo")
        for i in range(10):
            eo_image = self.eo_images[i]
            eo_image.save(
                os.path.join(
                    self.output_dir, folder, f"{basename}_eo{i}{self.file_extension}"
                )
            )

    @staticmethod
    def get_sample(x, modality):
        """

        :param x: floating point np array of shape (H, W, C=14) in [0.0, 1.0]
        :param modality: one of {'vv', 'vh', 'eo'}
        :return: PIL.Image.Image, or List[PIL.Image.Image] if modality == "eo"
        """
        sar_eps = 1e-9 + 1j * 1e-9

        if modality == "vh":
            x_vh = np.log10(
                np.abs(
                    np.complex128(
                        np.clip(x[..., 0], -1.0, 1.0)
                        + 1j * np.clip(x[..., 1], -1.0, 1.0)
                    )
                    + sar_eps
                )
            )
            sar_min = x_vh.min()
            sar_max = x_vh.max()
            sar_scale = 255.0 / (sar_max - sar_min)

            return Image.fromarray(np.uint8(sar_scale * (x_vh - sar_min)), "L")

        elif modality == "vv":
            x_vv = np.log10(
                np.abs(
                    np.complex128(
                        np.clip(x[..., 2], -1.0, 1.0)
                        + 1j * np.clip(x[..., 3], -1.0, 1.0)
                    )
                    + sar_eps
                )
            )
            sar_min = x_vv.min()
            sar_max = x_vv.max()
            sar_scale = 255.0 / (sar_max - sar_min)

            return Image.fromarray(np.uint8(sar_scale * (x_vv - sar_min)), "L")

        elif modality == "eo":
            eo_images = []

            eo_min = x[..., 4:].min()
            eo_max = x[..., 4:].max()
            eo_scale = 255.0 / (eo_max - eo_min)
            for c in range(4, 14):
                eo = Image.fromarray(
                    np.uint8(eo_scale * (np.clip(x[..., c], 0.0, 1.0) - eo_min)), "L"
                )
                eo_images.append(eo)

            return eo_images

        else:
            raise ValueError(
                f"modality must be one of ('vh', 'vv', 'eo'), received {modality}"
            )


class ExportMeter(Meter):
    def __init__(
        self, name, exporter, x_probe, y_probe=None, y_pred_probe=None, max_batches=None
    ):
        metric_args = [x_probe]
        if y_probe is not None:
            metric_args.append(y_probe)
        if y_pred_probe is not None:
            metric_args.append(y_pred_probe)
        super().__init__(name, lambda x: x, *metric_args)

        self.y_probe = y_probe
        self.y_pred_probe = y_pred_probe
        self.exporter = exporter
        self.max_batches = max_batches
        self.batches_exported = 0
        self.examples_exported = 0
        self.metric_args = metric_args

        if self.y_probe is not None:
            self.y_probe_idx = self.metric_args.index(self.y_probe)
        if self.y_pred_probe is not None:
            self.y_pred_probe_idx = self.metric_args.index(self.y_pred_probe)

    def measure(self, clear_values=True):
        self.is_ready(raise_error=True)
        batch_num, batch_data = self.arg_batch_indices[0], self.values[0]
        if self.max_batches and batch_num >= self.max_batches:
            return

        probe_variable = self.get_arg_names()[0]
        batch_size = batch_data.shape[0]
        examples_exported = 0
        for batch_idx in range(batch_size):
            export_kwargs = {}
            if self.y_probe is not None:
                export_kwargs["y"] = self.values[self.y_probe_idx][batch_idx]
            if self.y_pred_probe is not None:
                export_kwargs["y_pred"] = self.values[self.y_pred_probe_idx][batch_idx]
            self.exporter.export(
                batch_data[batch_idx],
                f"batch_{self.batches_exported}_ex_{self.examples_exported}_{probe_variable}",
                **export_kwargs,
            )
            examples_exported += 1
        self.batches_exported += 1
        if clear_values:
            self.clear()
        self.never_measured = False
