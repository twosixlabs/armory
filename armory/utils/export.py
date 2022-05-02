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


class SampleExporter:
    def __init__(self, base_output_dir, default_export_kwargs={}):
        self.base_output_dir = base_output_dir
        self.saved_batches = 0
        self.saved_samples = 0
        self.output_dir = None
        self.y_dict = {}
        self.default_export_kwargs = default_export_kwargs

    def export(
        self, x, x_adv=None, y=None, y_pred_adv=None, y_pred_clean=None, **kwargs
    ):
        export_kwargs = dict(
            list(self.default_export_kwargs.items()) + list(kwargs.items())
        )
        if self.saved_batches == 0:
            self._make_output_dir()

        self.y_dict[self.saved_samples] = {
            "y": y,
            "y_pred_clean": y_pred_clean,
            "y_pred_adv": y_pred_adv,
        }
        self._export(
            x=x,
            x_adv=x_adv,
            y=y,
            y_pred_adv=y_pred_adv,
            y_pred_clean=y_pred_clean,
            **export_kwargs,
        )

    @abc.abstractmethod
    def _export(
        self, x, x_adv=None, y=None, y_pred_adv=None, y_pred_clean=None, **kwargs
    ):
        raise NotImplementedError(
            f"_export() method should be defined for export class {self.__class__}"
        )

    @abc.abstractmethod
    def get_sample(self):
        raise NotImplementedError(
            f"get_sample() method should be defined for export class {self.__class__}"
        )

    def write(self):
        """Pickle the y_dict built up during each export() call.
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
            log.warning(
                f"Sample output directory {self.output_dir} already exists. Creating new directory"
            )
            self.output_dir = os.path.join(
                self.base_output_dir, f"saved_samples_{time.time()}"
            )
        os.mkdir(self.output_dir)


class ImageClassificationExporter(SampleExporter):
    def _export(self, x, x_adv=None, y=None, y_pred_adv=None, y_pred_clean=None):
        for i, x_i in enumerate(x):
            self._export_image(x_i, name="benign")

            # Export adversarial image x_adv_i if present
            if x_adv is not None:
                x_adv_i = x_adv[i]
                self._export_image(x_adv_i, name="adversarial")

            self.saved_samples += 1
        self.saved_batches += 1

    def _export_image(self, x_i, name="benign"):
        self.image = self.get_sample(x_i)
        self.image.save(
            os.path.join(self.output_dir, f"{self.saved_samples}_{name}.png")
        )
        if x_i.shape[-1] == 6:
            self.depth_image = self.get_sample(x_i[..., 3:])
            self.depth_image.save(
                os.path.join(self.output_dir, f"{self.saved_samples}_depth_{name}.png")
            )

    @staticmethod
    def get_sample(x_i):
        """

        :param x_i: floating point np array of shape (H, W, C) in [0.0, 1.0], where C = 1 (grayscale),
                3 (RGB), or 6 (RGB-Depth)
        :return: PIL.Image.Image
        """

        if x_i.min() < 0.0 or x_i.max() > 1.0:
            log.warning("Image out of expected range. Clipping to [0, 1].")

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


class ObjectDetectionExporter(ImageClassificationExporter):
    def __init__(self, base_output_dir, default_export_kwargs={}):
        super().__init__(base_output_dir, default_export_kwargs)
        self.ground_truth_boxes_coco_format = []
        self.benign_predicted_boxes_coco_format = []
        self.adversarial_predicted_boxes_coco_format = []

    def _export(
        self,
        x,
        x_adv=None,
        with_boxes=False,
        y=None,
        y_pred_adv=None,
        y_pred_clean=None,
        score_threshold=0.5,
        classes_to_skip=None,
    ):
        for i, x_i in enumerate(x):
            self._export_image(x_i, name="benign")

            if with_boxes:
                y_i = y[i] if y is not None else None
                y_i_pred_clean = y_pred_clean[i] if y_pred_clean is not None else None
                self._export_image(
                    x_i,
                    name="benign",
                    with_boxes=True,
                    y_i=y_i,
                    y_i_pred=y_i_pred_clean,
                    score_threshold=score_threshold,
                    classes_to_skip=classes_to_skip,
                )

            # Export adversarial image x_adv_i if present
            if x_adv is not None:
                x_adv_i = x_adv[i]
                self._export_image(x_adv_i, name="adversarial")
                if with_boxes:
                    y_i_pred_adv = y_pred_adv[i] if y_pred_adv is not None else None
                    self._export_image(
                        x_adv_i,
                        name="adversarial",
                        with_boxes=True,
                        y_i=y_i,
                        y_i_pred=y_i_pred_adv,
                        score_threshold=score_threshold,
                        classes_to_skip=classes_to_skip,
                    )

            self.saved_samples += 1
        self.saved_batches += 1

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
            x_i=x_i,
            with_boxes=True,
            y_i=y_i,
            y_i_pred=y_i_pred,
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
        x_i,
        with_boxes=False,
        y_i=None,
        y_i_pred=None,
        score_threshold=0.5,
        classes_to_skip=None,
    ):
        """
        :param x_i:  floating point np array of shape (H, W, C) in [0.0, 1.0], where C = 1 (grayscale),
                3 (RGB), or 6 (RGB-Depth)
        :param with_boxes: boolean indicating whether to display bounding boxes
        :param y_i: ground-truth label dict
        :param y_i_pred: predicted label dict
        :param score_threshold: float in [0, 1]; boxes with confidence > score_threshold are displayed
        :param classes_to_skip: List[Int] containing class ID's for which boxes should not be displayed
        :return: PIL.Image.Image
        """
        image = super().get_sample(x_i)
        if not with_boxes:
            return image

        if y_i is None and y_i_pred is None:
            raise TypeError("Both y_i and y_i_pred are None, but with_boxes is True")
        box_layer = ImageDraw.Draw(image)

        if y_i is not None:
            bboxes_true = y_i["boxes"]
            labels_true = y_i["labels"]

            for true_box, label in zip(bboxes_true, labels_true):
                if classes_to_skip is not None and label in classes_to_skip:
                    continue
                box_layer.rectangle(true_box, outline="red", width=2)

        if y_i_pred is not None:
            bboxes_pred = y_i_pred["boxes"][y_i_pred["scores"] > score_threshold]

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

    def _export(
        self, x, x_adv=None, y=None, y_pred_adv=None, y_pred_clean=None, **kwargs
    ):
        for i, x_i in enumerate(x):
            self._export_video(x_i, name="benign")

            if x_adv is not None:
                x_adv_i = x_adv[i]
                self._export_video(x_adv_i, name="adversarial")

            self.saved_samples += 1
        self.saved_batches += 1

    def _export_video(self, x_i, name="benign"):
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
                os.path.join(self.output_dir, folder, f"video_{name}.mp4"),
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
                os.path.join(self.output_dir, folder, f"frame_{n_frame:04d}_{name}.png")
            )

        ffmpeg_process.stdin.close()
        ffmpeg_process.wait()

    @staticmethod
    def get_sample(x_i):
        """

        :param x_i: floating point np array of shape (num_frames, H, W, C=3) in [0.0, 1.0]
        :return: List[PIL.Image.Image] of length equal to num_frames
        """
        if x_i.min() < 0.0 or x_i.max() > 1.0:
            log.warning("video out of expected range. Clipping to [0, 1]")

        pil_frames = []
        for n_frame, x_frame in enumerate(x_i):
            pixels = np.uint8(np.clip(x_frame, 0.0, 1.0) * 255.0)
            image = Image.fromarray(pixels, "RGB")
            pil_frames.append(image)
        return pil_frames


class VideoTrackingExporter(VideoClassificationExporter):
    def _export(
        self,
        x,
        x_adv=None,
        with_boxes=False,
        y=None,
        y_pred_adv=None,
        y_pred_clean=None,
    ):
        for i, x_i in enumerate(x):
            self._export_video(x_i, name="benign")

            if with_boxes:
                y_i = y[i] if y is not None else None
                y_i_pred_clean = y_pred_clean[i] if y_pred_clean is not None else None
                self._export_video(
                    x_i,
                    with_boxes=True,
                    y_i=y_i,
                    y_i_pred=y_i_pred_clean,
                    name="benign",
                )

            if x_adv is not None:
                x_adv_i = x_adv[i]
                self._export_video(x_adv_i, name="adversarial")
                if with_boxes:
                    y_i_pred_adv = y_pred_adv[i] if y_pred_adv is not None else None
                    self._export_video(
                        x_adv_i,
                        with_boxes=True,
                        y_i=y_i,
                        y_i_pred=y_i_pred_adv,
                        name="adversarial",
                    )

            self.saved_samples += 1
        self.saved_batches += 1

    def _export_video(
        self, x_i, with_boxes=False, y_i=None, y_i_pred=None, name="benign"
    ):
        if not with_boxes:
            super()._export_video(x_i, name=name)
            return

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
                os.path.join(self.output_dir, folder, f"video_{name}_with_boxes.mp4"),
                pix_fmt="yuv420p",
                vcodec="libx264",
                r=self.frame_rate,
            )
            .overwrite_output()
            .run_async(pipe_stdin=True, quiet=True)
        )

        self.frames_with_boxes = self.get_sample(
            x_i, with_boxes=True, y_i=y_i, y_i_pred=y_i_pred
        )
        for n_frame, frame in enumerate(self.frames_with_boxes):
            frame.save(
                os.path.join(
                    self.output_dir,
                    folder,
                    f"frame_{n_frame:04d}_{name}_with_boxes.png",
                )
            )
            pixels_with_boxes = np.array(frame)
            ffmpeg_process.stdin.write(pixels_with_boxes.tobytes())

        ffmpeg_process.stdin.close()
        ffmpeg_process.wait()

    def get_sample(self, x_i, with_boxes=False, y_i=None, y_i_pred=None):
        """

        :param x_i: floating point np array of shape (num_frames, H, W, C=3) in [0.0, 1.0]
        :param with_boxes: boolean indicating whether to display bounding boxes
        :param y_i: ground-truth label dict
        :param y_i_pred: predicted label dict
        :return: List[PIL.Image.Image] of length equal to num_frames
        """
        if not with_boxes:
            return super().get_sample(x_i)

        if y_i is None and y_i_pred is None:
            raise TypeError("Both y_i and y_pred are None, but with_boxes is True.")
        if x_i.min() < 0.0 or x_i.max() > 1.0:
            log.warning("video out of expected range. Clipping to [0,1]")

        pil_frames = []
        for n_frame, x_frame in enumerate(x_i):
            pixels = np.uint8(np.clip(x_frame, 0.0, 1.0) * 255.0)
            image = Image.fromarray(pixels, "RGB")
            box_layer = ImageDraw.Draw(image)
            if y_i is not None:
                bbox_true = y_i["boxes"][n_frame].astype("float32")
                box_layer.rectangle(bbox_true, outline="red", width=2)
            if y_i_pred is not None:
                bbox_pred = y_i_pred["boxes"][n_frame]
                box_layer.rectangle(bbox_pred, outline="white", width=2)
            pil_frames.append(image)

        return pil_frames


class AudioExporter(SampleExporter):
    def __init__(self, base_output_dir, sample_rate):
        self.sample_rate = sample_rate
        super().__init__(base_output_dir)

    def _export(
        self, x, x_adv=None, y=None, y_pred_adv=None, y_pred_clean=None, **kwargs
    ):
        for i, x_i in enumerate(x):
            self._export_audio(x_i, name="benign")

            if x_adv is not None:
                x_i_adv = x_adv[i]
                self._export_audio(x_i_adv, name="adversarial")

            self.saved_samples += 1
        self.saved_batches += 1

    def _export_audio(self, x_i, name="benign"):
        x_i_copy = deepcopy(x_i)

        if not np.isfinite(x_i_copy).all():
            posinf, neginf = 1, -1
            log.warning(
                f"audio vector has infinite values. Mapping nan to 0, -inf to {neginf}, inf to {posinf}."
            )
            x_i_copy = np.nan_to_num(x_i_copy, posinf=posinf, neginf=neginf)

        if x_i_copy.min() < -1.0 or x_i_copy.max() > 1.0:
            log.warning(
                "audio vector out of expected [-1, 1] range, normalizing by the max absolute value"
            )
            x_i_copy = x_i_copy / np.abs(x_i_copy).max()

        wavfile.write(
            os.path.join(self.output_dir, f"{self.saved_samples}_{name}.wav"),
            rate=self.sample_rate,
            data=x_i_copy,
        )

    @staticmethod
    def get_sample(x_i, dataset_context):
        """

        :param x_i: floating point np array of shape (sequence_length,) in [-1.0, 1.0]
        :param dataset_context: armory.data.datasets AudioContext object
        :return: int np array of shape (sequence_length, )
        """
        assert dataset_context.input_type == np.int64
        assert dataset_context.quantization == 2**15

        return np.clip(
            np.int16(x_i * dataset_context.quantization),
            dataset_context.input_min,
            dataset_context.input_max,
        )


class So2SatExporter(SampleExporter):
    def _export(
        self, x, x_adv=None, y=None, y_pred_adv=None, y_pred_clean=None, **kwargs
    ):

        for i, x_i in enumerate(x):
            self._export_so2sat_image(x_i, name="benign")

            if x_adv is not None:
                x_adv_i = x_adv[i]
                self._export_so2sat_image(x_adv_i, name="adversarial")

            self.saved_samples += 1
        self.saved_batches += 1

    def _export_so2sat_image(self, x_i, name="benign"):
        folder = str(self.saved_samples)
        os.makedirs(os.path.join(self.output_dir, folder), exist_ok=True)

        self.vh_image = self.get_sample(x_i, modality="vh")
        self.vh_image.save(os.path.join(self.output_dir, folder, f"vh_{name}.png"))

        self.vv_image = self.get_sample(x_i, modality="vv")
        self.vv_image.save(os.path.join(self.output_dir, folder, f"vv_{name}.png"))

        self.eo_images = self.get_sample(x_i, modality="eo")
        for i in range(10):
            eo_image = self.eo_images[i]
            eo_image.save(os.path.join(self.output_dir, folder, f"eo{i}_{name}.png"))

    @staticmethod
    def get_sample(x_i, modality):
        """

        :param x_i: floating point np array of shape (H, W, C=14) in [0.0, 1.0]
        :param modality: one of {'vv', 'vh', 'eo'}
        :return: PIL.Image.Image, or List[PIL.Image.Image] if modality == "eo"
        """
        sar_eps = 1e-9 + 1j * 1e-9

        if modality == "vh":
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

            return Image.fromarray(np.uint8(sar_scale * (x_vh - sar_min)), "L")

        elif modality == "vv":
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

            return Image.fromarray(np.uint8(sar_scale * (x_vv - sar_min)), "L")

        elif modality == "eo":
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

        else:
            raise ValueError(
                f"modality must be one of ('vh', 'vv', 'eo'), received {modality}"
            )
