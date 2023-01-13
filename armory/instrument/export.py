import abc
from copy import deepcopy
import json
import os
import pickle

from PIL import Image, ImageDraw
import ffmpeg
import numpy as np
from scipy.io import wavfile

from armory.instrument import Meter
from armory.logs import log


class SampleExporter:
    def __init__(self, output_dir, default_export_kwargs={}):
        self.output_dir = output_dir
        base_dir = os.path.dirname(self.output_dir)
        if base_dir and not os.path.exists(base_dir):
            raise ValueError(f"Directory {base_dir} does not exist")
        self.default_export_kwargs = default_export_kwargs

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

    def _make_output_dir(self):
        os.mkdir(self.output_dir)


class ImageClassificationExporter(SampleExporter):
    def __init__(self, output_dir, default_export_kwargs={}):
        super().__init__(
            output_dir=output_dir, default_export_kwargs=default_export_kwargs
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
        image = Image.fromarray(
            np.uint8(np.round(np.clip(x_i_mode, 0.0, 1.0) * 255.0)), mode
        )
        return image


class ObjectDetectionExporter(ImageClassificationExporter):
    def __init__(self, output_dir, default_export_kwargs={}):
        super().__init__(output_dir, default_export_kwargs)
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
            raise TypeError("Both y and y_pred are None, but with_boxes is True")
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


class DApricotExporter(ObjectDetectionExporter):
    def _export(
        self,
        x,
        basename,
        with_boxes=False,
        y=None,
        y_pred=None,
        score_threshold=0.5,
        classes_to_skip=[12],  # class of green-screen
    ):
        x_angle_1 = x[0]
        x_angle_2 = x[1]
        x_angle_3 = x[2]

        if with_boxes:
            y_pred_angle_1 = deepcopy(y_pred[0])
            y_pred_angle_1["boxes"] = self.convert_boxes_tf_to_torch(
                x_angle_1, y_pred_angle_1["boxes"]
            )
            super()._export(
                x_angle_1,
                f"{basename}_angle_1",
                with_boxes=True,
                y_pred=y_pred_angle_1,
                score_threshold=score_threshold,
                classes_to_skip=classes_to_skip,
            )

            y_pred_angle_2 = deepcopy(y_pred[1])
            y_pred_angle_2["boxes"] = self.convert_boxes_tf_to_torch(
                x_angle_2, y_pred_angle_2["boxes"]
            )

            super()._export(
                x_angle_2,
                f"{basename}_angle_2",
                with_boxes=True,
                y_pred=y_pred_angle_2,
                score_threshold=score_threshold,
                classes_to_skip=classes_to_skip,
            )

            y_pred_angle_3 = deepcopy(y_pred[2])
            y_pred_angle_3["boxes"] = self.convert_boxes_tf_to_torch(
                x_angle_3, y_pred_angle_3["boxes"]
            )
            super()._export(
                x_angle_3,
                f"{basename}_angle_3",
                with_boxes=True,
                y_pred=y_pred_angle_3,
                score_threshold=score_threshold,
                classes_to_skip=classes_to_skip,
            )
        else:
            super()._export(x_angle_1, f"{basename}_angle_1")
            super()._export(x_angle_2, f"{basename}_angle_2")
            super()._export(x_angle_3, f"{basename}_angle_3")

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
    def __init__(self, output_dir, frame_rate, default_export_kwargs={}):
        super().__init__(output_dir, default_export_kwargs=default_export_kwargs)
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
            pixels = np.uint8(np.round(np.clip(x_frame, 0.0, 1.0) * 255.0))
            image = Image.fromarray(pixels, "RGB")
            pil_frames.append(image)
        return pil_frames


class VideoTrackingExporter(VideoClassificationExporter):
    def convert_y_from_mot_format(self, y):
        # multi-object tracking format:
        # <timestep> <object_id> <bbox top-left x> <bbox top-left y> <bbox width> <bbox height> <confidence_score=1> <class_id> <visibility=1>
        # desired format:
        # {'boxes': {<timestep>:[[x0, y0, x1, y1]...]}}

        if y.ndim == 1:
            y = np.expand_dims(y, 0)
        y_dict = {
            int(timestep - 1): [] for timestep in set(y[:, 0])
        }  # 0-index timesteps

        for pred in y:
            # TODO filter out boxes with low confidence?
            timestep = int(pred[0] - 1)
            x0, y0 = pred[2], pred[3]
            x1, y1 = pred[2] + pred[4], pred[3] + pred[5]
            y_dict[timestep].append([x0, y0, x1, y1])

        for timestep in y_dict:
            y_dict[timestep] = np.array(y_dict[timestep]).astype("float32")

        return {"boxes": y_dict}

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

            # reformat y and y_pred if in MOT format
            if isinstance(y, np.ndarray):
                y = self.convert_y_from_mot_format(y)
            if isinstance(y_pred, np.ndarray):
                y_pred = self.convert_y_from_mot_format(y_pred)

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
        :param y: either (1) a dict with "boxes" key mapping to dict with keys for each frame index,
                  each frame index key mapping to a numpy array of shape (num_boxes_in_frame, 4) or
                  (2) coco-formatted List[dict] of length # total boxes across all frames, where
                  each dict has an "image_id" key and a "bbox" key mapping to a sequence of
                  [x1, y1, width, height]
        :param y_pred: " "
        :return: List[PIL.Image.Image] of length equal to num_frames
        """
        if not with_boxes:
            return super().get_sample(x)

        if y is None and y_pred is None:
            raise TypeError("Both y and y_pred are None, but with_boxes is True.")
        if x.min() < 0.0 or x.max() > 1.0:
            log.warning("video out of expected range. Clipping to [0,1]")

        pil_frames = []
        for n_frame, x_frame in enumerate(x):
            pixels = np.uint8(np.round(np.clip(x_frame, 0.0, 1.0) * 255.0))
            image = Image.fromarray(pixels, "RGB")
            box_layer = ImageDraw.Draw(image)

            for annotation, box_color in zip([y, y_pred], ["red", "white"]):
                if annotation is not None:
                    if isinstance(annotation, dict):
                        bboxes_true = annotation["boxes"][n_frame].astype("float32")
                        if bboxes_true.ndim == 1:
                            box_layer.rectangle(bboxes_true, outline=box_color, width=2)
                        else:
                            for bbox_true in bboxes_true:
                                box_layer.rectangle(
                                    bbox_true, outline=box_color, width=2
                                )
                    elif isinstance(annotation, list):
                        bboxes_true = [
                            box_anno["bbox"]
                            for box_anno in annotation
                            if box_anno["image_id"] == n_frame
                        ]
                        for bbox_true in bboxes_true:
                            box_x, box_y, box_width, box_height = bbox_true
                            box_layer.rectangle(
                                (box_x, box_y, box_x + box_width, box_y + box_height),
                                outline=box_color,
                                width=2,
                            )
                    else:
                        raise TypeError(f"Found unexpected type {type(annotation)}")
            pil_frames.append(image)

        return pil_frames


class AudioExporter(SampleExporter):
    def __init__(self, output_dir, sample_rate):
        self.sample_rate = sample_rate
        self.file_extension = ".wav"
        super().__init__(output_dir)

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
    def __init__(self, output_dir, default_export_kwargs={}):
        super().__init__(
            output_dir=output_dir, default_export_kwargs=default_export_kwargs
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

            return Image.fromarray(
                np.uint8(np.round(sar_scale * (x_vh - sar_min))), "L"
            )

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

            return Image.fromarray(
                np.uint8(np.round(sar_scale * (x_vv - sar_min))), "L"
            )

        elif modality == "eo":
            eo_images = []

            eo_min = x[..., 4:].min()
            eo_max = x[..., 4:].max()
            eo_scale = 255.0 / (eo_max - eo_min)
            for c in range(4, 14):
                eo = Image.fromarray(
                    np.uint8(
                        np.round(eo_scale * (np.clip(x[..., c], 0.0, 1.0) - eo_min))
                    ),
                    "L",
                )
                eo_images.append(eo)

            return eo_images

        else:
            raise ValueError(
                f"modality must be one of ('vh', 'vv', 'eo'), received {modality}"
            )


class ExportMeter(Meter):
    def __init__(
        self,
        name,
        exporter,
        x_probe,
        y_probe=None,
        y_pred_probe=None,
        max_batches=None,
        overwrite_mode="increment",
    ):
        """
        :param name (string): name given to ExportMeter
        :param exporter (SampleExporter): sample exporter object
        :param x_probe (string): name of x probe e.g. "scenario.x"
        :param y_probe (string or None): name of y probe, if applicable. E.g. "scenario.y"
        :param y_pred_probe (string or None): name of y_pred_probe, if applicable. E.g. "scenario.y_pred"
        :param max_batches (int or None): maximum number of batches to export
        :param overwrite_mode (string): one of ('increment', 'overwrite'). Whether to overwrite existing
                files or increment filename with an appended underscore and number
        """
        if overwrite_mode not in ["increment", "overwrite"]:
            raise ValueError(
                f"overwrite_mode must be one of ('increment', 'overwrite'). Received {overwrite_mode}"
            )
        self.overwrite_mode = overwrite_mode
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
        self.metric_args = metric_args

        if self.y_probe is not None:
            self.y_probe_idx = self.metric_args.index(self.y_probe)
        if self.y_pred_probe is not None:
            self.y_pred_probe_idx = self.metric_args.index(self.y_pred_probe)

        self.sample_id_counter = {}

    def measure(self, clear_values=True):
        self.is_ready(raise_error=True)
        batch_num, batch_data = self.arg_batch_indices[0], self.values[0]
        if self.max_batches is not None and batch_num >= self.max_batches:
            return

        probe_variable = self.get_arg_names()[0]
        batch_size = batch_data.shape[0]
        for batch_idx in range(batch_size):
            export_kwargs = {}
            if self.y_probe is not None:
                export_kwargs["y"] = self.values[self.y_probe_idx][batch_idx]
            if self.y_pred_probe is not None:
                export_kwargs["y_pred"] = self.values[self.y_pred_probe_idx][batch_idx]

            sample_id = f"batch_{batch_num}_ex_{batch_idx}_{probe_variable}"
            sample_id_count = self.sample_id_counter.get(sample_id, 0)
            if sample_id_count > 0 and self.overwrite_mode == "increment":
                basename = f"{sample_id}_{sample_id_count}"
            else:
                self.sample_id_counter[sample_id] = 0
                basename = sample_id

            self.sample_id_counter[sample_id] += 1
            self.exporter.export(
                batch_data[batch_idx],
                basename,
                **export_kwargs,
            )
        if clear_values:
            self.clear()
        self.never_measured = False

    def finalize(self):
        if self.never_measured:
            unset = [arg for arg, i in self.arg_index.items() if not self.values_set[i]]
            if unset:
                log.warning(
                    f"Meter '{self.name}' was never measured. "
                    f"The following args were never set: {unset}"
                )


class PredictionMeter(Meter):
    """
    Meter that keeps track of y, y_pred_clean, y_pred_adv and pickles these results to a dictionary
    that gets saved in the Armory scenario output directory. Each key in the dictionary corresponds
    to the dataset example index, which maps to "y", "y_pred_clean", and "y_pred_adv" values.
    """

    def __init__(
        self,
        name,
        output_dir,
        y_probe=None,
        y_pred_clean_probe=None,
        y_pred_adv_probe=None,
        max_batches=None,
    ):
        idx = 0
        metric_args = []
        if y_probe is not None:
            metric_args.append(y_probe)
            self.y_probe_idx = idx
            idx += 1
        if y_pred_clean_probe is not None:
            metric_args.append(y_pred_clean_probe)
            self.y_pred_clean_probe_idx = idx
            idx += 1
        if y_pred_adv_probe is not None:
            metric_args.append(y_pred_adv_probe)
            self.y_pred_adv_probe_idx = idx

        super().__init__(name, lambda x: x, *metric_args)
        if not metric_args:
            log.warning(f"{self.name} was constructed with all probes set to None")

        self.output_dir = output_dir
        self.y_probe = y_probe
        self.y_pred_clean_probe = y_pred_clean_probe
        self.y_pred_adv_probe = y_pred_adv_probe
        self.max_batches = max_batches
        self.examples_saved = 0
        self.y_dict = {}

    def measure(self, clear_values=True):
        self.is_ready(raise_error=True)
        batch_num = self.arg_batch_indices[0]
        batch_size = len(self.values[0])

        if (
            self.max_batches is not None and batch_num >= self.max_batches
        ) or not self.values:
            return

        y = [None] * batch_size
        y_pred_clean = [None] * batch_size
        y_pred_adv = [None] * batch_size

        if self.y_probe is not None:
            y = self.values[self.y_probe_idx]
        if self.y_pred_clean_probe is not None:
            y_pred_clean = self.values[self.y_pred_clean_probe_idx]
        if self.y_pred_adv_probe is not None:
            y_pred_adv = self.values[self.y_pred_adv_probe_idx]

        for i in range(batch_size):
            y_i = y[i]
            y_i_pred_clean = y_pred_clean[i]
            y_i_pred_adv = y_pred_adv[i]
            self.y_dict[self.examples_saved] = {
                "y": y_i,
                "y_pred": y_i_pred_clean,
                "y_pred_adv": y_i_pred_adv,
            }
            self.examples_saved += 1
        if clear_values:
            self.clear()
        self.never_measured = False

    def finalize(self):
        if self.never_measured:
            unset = [arg for arg, i in self.arg_index.items() if not self.values_set[i]]
            if unset:
                log.warning(
                    f"Meter '{self.name}' was never measured. "
                    f"The following args were never set: {unset}"
                )

        if self.examples_saved > 0:
            with open(os.path.join(self.output_dir, "predictions.pkl"), "wb") as f:
                pickle.dump(self.y_dict, f)


class CocoBoxFormatMeter(Meter):
    """
    Meter that keeps track of object detection predicted boxes and saves them
    as COCO-formatted JSON files.
    """

    def __init__(
        self,
        name,
        output_dir,
        y_probe,
        y_pred_clean_probe=None,
        y_pred_adv_probe=None,
        max_batches=None,
    ):
        metric_args = [y_probe]
        self.y_probe_idx = 0
        self.y_pred_clean_probe_idx = None
        self.y_pred_adv_probe_idx = None

        idx = 1
        if y_pred_clean_probe is not None:
            metric_args.append(y_pred_clean_probe)
            self.y_pred_clean_probe_idx = idx
            idx += 1
        if y_pred_adv_probe is not None:
            metric_args.append(y_pred_adv_probe)
            self.y_pred_adv_probe_idx = idx

        super().__init__(name, lambda x: x, *metric_args)
        if not metric_args:
            log.warning(f"{self.name} was constructed with all probes set to None")

        self.output_dir = output_dir
        self.y_probe = y_probe
        self.y_pred_clean_probe = y_pred_clean_probe
        self.y_pred_adv_probe = y_pred_adv_probe
        self.max_batches = max_batches
        self.y_boxes_coco_format = []
        self.y_pred_clean_boxes_coco_format = []
        self.y_pred_adv_boxes_coco_format = []

    def measure(self, clear_values=True):
        self.is_ready(raise_error=True)
        batch_num = self.arg_batch_indices[0]
        batch_size = len(self.values[0])

        if (
            self.max_batches is not None and batch_num >= self.max_batches
        ) or not self.values:
            return

        for ex_idx in range(batch_size):
            # All boxes for a given example have the same image id
            image_id = int(
                self.values[self.y_probe_idx][ex_idx].get("image_id").flatten()[0]
            )
            for probe_idx, box_list in [
                (self.y_probe_idx, self.y_boxes_coco_format),
                (self.y_pred_clean_probe_idx, self.y_pred_clean_boxes_coco_format),
                (self.y_pred_adv_probe_idx, self.y_pred_adv_boxes_coco_format),
            ]:
                if probe_idx is not None:
                    boxes = self.values[probe_idx][ex_idx]
                    boxes_coco_format = self.get_coco_formatted_bounding_box_data(
                        boxes, image_id=image_id
                    )
                    box_list.extend(boxes_coco_format)

    def finalize(self):
        if self.never_measured:
            unset = [arg for arg, i in self.arg_index.items() if not self.values_set[i]]
            if unset:
                log.warning(
                    f"Meter '{self.name}' was never measured. "
                    f"The following args were never set: {unset}"
                )
        for box_list, name in [
            (self.y_boxes_coco_format, "ground_truth"),
            (self.y_pred_clean_boxes_coco_format, "benign_predicted"),
            (self.y_pred_adv_boxes_coco_format, "adversarial_predicted"),
        ]:
            if len(box_list) > 0:
                json.dump(
                    box_list,
                    open(
                        os.path.join(self.output_dir, f"{name}_boxes_coco_format.json"),
                        "w",
                    ),
                )

    def get_coco_formatted_bounding_box_data(
        self, y, image_id=None, score_threshold=0.5, classes_to_skip=None
    ):
        """
        :param y: ground-truth label dict or predicted label dict
        :param score_threshold: float in [0, 1]; predicted boxes with confidence > score_threshold are exported
        :param image_id: int or None. This key exists in ground-truth boxes but not predicted boxes, in which case it should be passed in
        :param classes_to_skip: List[Int] containing class ID's for which boxes should not be exported
        :return: List of dictionaries, containing coco-formatted bounding box data for ground truth and predicted labels
        """

        boxes_coco_format = []

        # "image_id" key will exist for ground-truth but not predicted boxes
        if image_id is None:
            image_id = y["image_id"][0]  # All boxes in y are for the same image
        elif not isinstance(image_id, int):
            raise ValueError(f"Expected an int for image_id, received {type(image_id)}")

        bboxes = y["boxes"]
        cat_ids = y["labels"]

        # "scores" key will exist for predicted boxes, but not ground-truth
        scores = y.get("scores")
        if scores is not None:
            bboxes = bboxes[scores > score_threshold]
            cat_ids = cat_ids[scores > score_threshold]
            scores = scores[scores > score_threshold]

        for i, (box, cat_id) in enumerate(zip(bboxes, cat_ids)):
            if classes_to_skip is not None and cat_id in classes_to_skip:
                continue
            xmin, ymin, xmax, ymax = box
            coco_box = {
                "image_id": int(image_id),
                "category_id": int(cat_id),
                "bbox": [int(xmin), int(ymin), int(xmax - xmin), int(ymax - ymin)],
            }
            if scores is not None:
                coco_box["score"] = float(scores[i])
            boxes_coco_format.append(coco_box)

        return boxes_coco_format
