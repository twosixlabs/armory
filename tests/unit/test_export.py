import pytest
import numpy as np
import PIL

from armory.utils.export import (
    ImageClassificationExporter,
    ObjectDetectionExporter,
    VideoClassificationExporter,
    VideoTrackingExporter,
    So2SatExporter,
)

# Mark all tests in this file as `unit`
pytestmark = pytest.mark.unit


# Image Classification test input
random_img = np.random.rand(32, 32, 3)

# Object Detection test inputs
obj_det_y_i = {
    "labels": np.array([1.0]),
    "boxes": np.array([[0.0, 0.0, 1.0, 1.0]]).astype(np.float32),
    "image_id": np.array([1]),
}
obj_det_y_i_pred = {
    "scores": np.array([1.0]),
    "labels": np.array([1.0]),
    "boxes": np.array([[0.0, 0.0, 1.0, 1.0]]).astype(np.float32),
}

# Video Test Inputs
num_frames = 10
random_video = np.random.rand(num_frames, 32, 32, 3)
video_tracking_y = {"boxes": np.ones((num_frames, 4)).astype(np.float32)}
video_tracking_y_pred = {"boxes": np.ones((num_frames, 4)).astype(np.float32)}

# So2Sat Test Input
random_so2sat_img = np.random.rand(32, 32, 14)


@pytest.mark.parametrize(
    "exporter_class, class_kwargs, input_array, fn_kwargs, expected_output_type",
    [
        (ImageClassificationExporter, {}, random_img, {}, PIL.Image.Image),
        (ObjectDetectionExporter, {}, random_img, {}, PIL.Image.Image),
        (
            ObjectDetectionExporter,
            {},
            random_img,
            {"with_boxes": True, "y_i": obj_det_y_i, "y_i_pred": obj_det_y_i_pred},
            PIL.Image.Image,
        ),
        (
            ObjectDetectionExporter,
            {},
            random_img,
            {"with_boxes": True, "y_i": obj_det_y_i},
            PIL.Image.Image,
        ),
        (
            ObjectDetectionExporter,
            {},
            random_img,
            {"with_boxes": True, "y_i_pred": obj_det_y_i_pred},
            PIL.Image.Image,
        ),
        (VideoClassificationExporter, {"frame_rate": 10}, random_video, {}, list),
        (VideoTrackingExporter, {"frame_rate": 10}, random_video, {}, list),
        (
            VideoTrackingExporter,
            {"frame_rate": 10},
            random_video,
            {
                "with_boxes": True,
                "y_i": video_tracking_y,
                "y_i_pred": video_tracking_y_pred,
            },
            list,
        ),
        (
            VideoTrackingExporter,
            {"frame_rate": 10},
            random_video,
            {"with_boxes": True, "y_i_pred": video_tracking_y_pred},
            list,
        ),
        (
            VideoTrackingExporter,
            {"frame_rate": 10},
            random_video,
            {"with_boxes": True, "y_i": video_tracking_y},
            list,
        ),
        (So2SatExporter, {}, random_so2sat_img, {"modality": "vh"}, PIL.Image.Image),
        (So2SatExporter, {}, random_so2sat_img, {"modality": "vv"}, PIL.Image.Image),
        (So2SatExporter, {}, random_so2sat_img, {"modality": "eo"}, list),
    ],
)
def test_exporter(
    exporter_class, class_kwargs, input_array, fn_kwargs, expected_output_type, tmp_path
):
    exporter = exporter_class(base_output_dir=tmp_path, **class_kwargs)
    sample = exporter.get_sample(input_array, **fn_kwargs)
    assert isinstance(sample, expected_output_type)

    # For object detection, check that coco annotations can be created
    if exporter_class == ObjectDetectionExporter:
        y_i, y_i_pred = fn_kwargs.get("y_i", None), fn_kwargs.get("y_i_pred", None)
        if y_i is not None:
            box_data_lists = exporter.get_coco_formatted_bounding_box_data(
                y_i, y_i_pred
            )
            for box_list in box_data_lists:
                assert isinstance(box_list, list)

    # For video scenarios, check that the list contains num_frames elements and each is a PIL Image
    if exporter_class in [VideoClassificationExporter, VideoTrackingExporter]:
        assert len(sample) == num_frames
        for i in sample:
            assert isinstance(i, PIL.Image.Image)

    # For the so2sat eo test, check that the list contains 10 elements and each is a PIL Image
    if exporter_class == So2SatExporter and fn_kwargs["modality"] == "eo":
        assert len(sample) == 10
        for i in sample:
            assert isinstance(i, PIL.Image.Image)
