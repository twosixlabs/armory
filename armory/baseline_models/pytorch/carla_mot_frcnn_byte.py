"""
Modified ByteTrack (https://arxiv.org/pdf/2110.06864.pdf) by replacing YOLOX object
detector with a PyTorch Faster-RCNN Resnet50-FPN object detector
"""
import dataclasses
from typing import List, Optional

from art.estimators.object_detection import PyTorchFasterRCNN
import numpy as np
import torch
from torchvision import models
from yolox.tracker.byte_tracker import (  # clone from https://github.com/ifzhang/ByteTrack
    BYTETracker,
)

from armory.data.adversarial_datasets import mot_array_to_coco

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclasses.dataclass
class BYTETrackerArgs:
    track_thresh: float
    track_buffer: int
    mot20: bool
    match_thresh: float


class PyTorchTracker(PyTorchFasterRCNN):
    """
    This wrapper adds BYTE tracking to the PyTorchFasterRCNN outputs
    """

    def __init__(
        self,
        model,
        clip_values,
        channels_first,
        coco_format=False,
        **wrapper_kwargs,
    ):
        BYTE_kwargs = wrapper_kwargs.pop("BYTE_kwargs", {})

        track_thresh = BYTE_kwargs.pop("track_thresh", 0.65)
        track_buffer = BYTE_kwargs.pop("track_buffer", 20)
        match_thresh = BYTE_kwargs.pop("match_thresh", 0.5)
        self.frame_rate = BYTE_kwargs.pop("frame_rate", 30)

        self.tracked_classes = wrapper_kwargs.pop("tracked_classes", ["pedestrian"])
        self.tracked_classes_map = {"pedestrian": 1, "vehicle": 2}

        self.tracked_classes.sort()
        if self.tracked_classes != ["pedestrian"]:
            raise ValueError('tracked_classes must be ["pedestrian"]')

        self.conf_thresh = wrapper_kwargs.pop("conf_thresh", 0.1)
        self.nms_thresh = wrapper_kwargs.pop("nms_thresh", 0.5)

        self.coco_format = coco_format

        self.tracker_args = BYTETrackerArgs(
            track_thresh=track_thresh,
            track_buffer=track_buffer,
            mot20=True,
            match_thresh=match_thresh,
        )

        super().__init__(
            model=model,
            clip_values=clip_values,
            channels_first=channels_first,
            **wrapper_kwargs,
        )

    def predict_object_detector(self, x, **kwargs):
        """
        Perform object detection prediction for a batch of inputs.

        :param x: Samples of shape (nb_samples, height, width, nb_channels) representing one video.
        :return: Predictions of format `List[Dict[str, np.ndarray]]`, one for each input image. The fields of the Dict
                 are as follows:
                 - boxes [N, 4]: the boxes in [x1, y1, x2, y2] format, with 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H.
                 - labels [N]: the labels for each image
                 - scores [N]: the scores or each prediction.
        """
        return super().predict(x, **kwargs)

    def predict(self, x):
        """
        Perform tracking prediction for a batch of inputs by performing object detection, updating Kalman filters, and outputing filter predictions.

        :param x: Samples of shape (n_batch, nb_samples, height, width, nb_channels) representing one video.  n_batch is assumed to be 1.
        :return: tracker detections as 2D ndarray, where each row has format:
                <timestep> <object_id> <bbox top-left x> <bbox top-left y> <bbox width> <bbox height> <confidence_score> <-1> <-1> <-1>
        """

        import torchvision  # lgtm [py/repeated-import]

        self._model.eval()

        # Apply preprocessing
        x, _ = self._apply_preprocessing(x[0], y=None, fit=False)

        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        image_tensor_list: List[np.ndarray] = []

        if self.clip_values is not None:
            norm_factor = self.clip_values[1]
        else:
            norm_factor = 1.0
        for i in range(x.shape[0]):
            image_tensor_list.append(transform(x[i] / norm_factor).to(self.device))
        predictions = self._model(image_tensor_list)

        # Recreate the tracker to reset its tracks
        tracker = BYTETracker(self.tracker_args, frame_rate=self.frame_rate)

        results = []
        # Iterate over the batch (or timestep) of predictions and update tracker
        for frame_id, pred in enumerate(predictions):
            with torch.no_grad():
                boxes = pred["boxes"]
                scores = pred["scores"]
                labels = pred["labels"]

                # Keep only predictions associated with tracked classes and whose scores is above threshold
                for tc in self.tracked_classes:
                    cls_id = self.tracked_classes_map[tc]
                    boxes_c = boxes[labels == cls_id].clone()
                    scores_c = scores[labels == cls_id].clone()
                    labels_c = labels[labels == cls_id].clone()

                    boxes_c = boxes_c[scores_c >= self.conf_thresh]
                    labels_c = labels_c[scores_c >= self.conf_thresh]
                    scores_c = scores_c[scores_c >= self.conf_thresh]

                    # Perform non-maximum suppression to remove redundant bounding boxes
                    # and reformat prediction as required by tracker, which is
                    # (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
                    nms_out_index = torchvision.ops.batched_nms(
                        boxes_c,
                        scores_c,
                        labels_c,
                        self.nms_thresh,
                    )

                    detections = torch.cat(
                        (
                            boxes_c,
                            torch.unsqueeze(scores_c, 1),
                            torch.ones(len(scores_c), 1).to(DEVICE),
                            torch.unsqueeze(labels_c, 1),
                        ),
                        1,
                    )
                    detections = detections[nms_out_index]

                    # Update tracker
                    if detections.size(0):
                        online_targets = tracker.update(
                            detections, x.shape[1:3], x.shape[1:3]
                        )
                        online_tlwhs = []
                        online_ids = []
                        online_scores = []
                        for t in online_targets:
                            tlwh = t.tlwh
                            tid = t.track_id
                            online_tlwhs.append(tlwh)
                            online_ids.append(tid)
                            online_scores.append(t.score)

                        # save results
                        results.append(
                            (
                                [
                                    frame_id for _ in range(len(online_ids))
                                ],  # Use 0-based index for MOT frame
                                online_ids,
                                online_tlwhs,
                                online_scores,
                                [cls_id for _ in range(len(online_ids))],
                                [
                                    1 for _ in range(len(online_ids))
                                ],  # visibility; not used
                            )
                        )

        # Format tracker output to format required by metrics calculation, namely,
        # tracker detections are given as 2D NDArrays with shape = (M, 9). Each row is a detection whose format is:
        # <timestep> <object_id> <bbox top-left x> <bbox top-left y> <bbox width> <bbox height> <confidence_score> <class_id> <visibility=1>
        output = [
            [f, i, *b, s, c, v]
            for result in results
            for [f, i, b, s, c, v] in zip(*result)
        ]
        output = np.asarray(output).astype(np.float32)
        output = np.expand_dims(output, 0)

        if self.coco_format:
            output = mot_array_to_coco(output)
        return output


# NOTE: PyTorchFasterRCNN expects numpy input, not torch.Tensor input
def get_art_model(
    model_kwargs: dict, wrapper_kwargs: dict, weights_path: Optional[str] = None
) -> PyTorchFasterRCNN:

    if weights_path:
        assert model_kwargs.get("num_classes", None) == 2, (
            "model trained on CARLA data outputs predictions for 2 classes, "
            "set model_kwargs['num_classes'] to 2."
        )
        assert not model_kwargs.get("pretrained", False), (
            "model trained on CARLA data should not use COCO-pretrained weights, set "
            "model_kwargs['pretrained'] to False."
        )

    model = models.detection.fasterrcnn_resnet50_fpn(**model_kwargs)
    model.to(DEVICE)

    if weights_path:
        checkpoint = torch.load(weights_path, map_location=DEVICE)
        model.load_state_dict(checkpoint)

    wrapped_model = PyTorchTracker(
        model,
        clip_values=(0.0, 1.0),
        channels_first=False,
        **wrapper_kwargs,
    )
    return wrapped_model
