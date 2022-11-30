import os
import subprocess

import PIL
import numpy as np
import pytest

from armory.art_experimental.attacks.carla_obj_det_utils import (
    linear_depth_to_rgb,
    linear_to_log,
    log_to_linear,
    rgb_depth_to_linear,
)
from armory.instrument import get_hub, get_probe
from armory.instrument.export import (
    CocoBoxFormatMeter,
    ExportMeter,
    ImageClassificationExporter,
    ObjectDetectionExporter,
    PredictionMeter,
    So2SatExporter,
    VideoClassificationExporter,
    VideoTrackingExporter,
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
            {"with_boxes": True, "y": obj_det_y_i, "y_pred": obj_det_y_i_pred},
            PIL.Image.Image,
        ),
        (
            ObjectDetectionExporter,
            {},
            random_img,
            {"with_boxes": True, "y": obj_det_y_i},
            PIL.Image.Image,
        ),
        (
            ObjectDetectionExporter,
            {},
            random_img,
            {"with_boxes": True, "y_pred": obj_det_y_i_pred},
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
                "y": video_tracking_y,
                "y_pred": video_tracking_y_pred,
            },
            list,
        ),
        (
            VideoTrackingExporter,
            {"frame_rate": 10},
            random_video,
            {"with_boxes": True, "y_pred": video_tracking_y_pred},
            list,
        ),
        (
            VideoTrackingExporter,
            {"frame_rate": 10},
            random_video,
            {"with_boxes": True, "y": video_tracking_y},
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
    exporter = exporter_class(tmp_path, **class_kwargs)
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


hub = get_hub()
BATCH_SIZE = 2
NUM_BATCHES = 2
IMAGE_BATCH = np.random.rand(BATCH_SIZE, 32, 32, 3)


@pytest.mark.parametrize(
    "name, x, exporter_class, x_probe, max_batches, overwrite_mode",
    [
        (
            "max_batches=None, overwrite_mode=increment",
            IMAGE_BATCH,
            ImageClassificationExporter,
            "scenario.x",
            None,
            "increment",
        ),
        (
            "max_batches=None, overwrite_mode=overwrite",
            IMAGE_BATCH,
            ImageClassificationExporter,
            "scenario.x",
            None,
            "overwrite",
        ),
        (
            "max_batches=1, overwrite_mode=increment",
            IMAGE_BATCH,
            ImageClassificationExporter,
            "scenario.x",
            1,
            "increment",
        ),
        (
            "max_batches=1, overwrite_mode=overwrite",
            IMAGE_BATCH,
            ImageClassificationExporter,
            "scenario.x",
            1,
            "increment",
        ),
    ],
)
def test_export_meters(
    name,
    x,
    exporter_class,
    x_probe,
    max_batches,
    overwrite_mode,
    tmp_path,
):
    exporter = exporter_class(tmp_path)
    export_meter = ExportMeter(
        name,
        exporter,
        x_probe,
        y_probe=None,
        y_pred_probe=None,
        max_batches=max_batches,
        overwrite_mode=overwrite_mode,
    )
    is_incrementing = overwrite_mode == "increment"
    hub.connect_meter(export_meter, use_default_writers=False)
    probe = get_probe("scenario")
    for i in range(NUM_BATCHES):
        hub.set_context(batch=i)
        probe.update(x=x)
        probe.update(x=x)  # calling a second time to test overwrite_mode

    num_samples_exported = len(os.listdir(tmp_path))
    if max_batches is None:
        num_samples_expected = BATCH_SIZE * NUM_BATCHES * (is_incrementing + 1)
    else:
        num_samples_expected = (
            BATCH_SIZE * min(max_batches, NUM_BATCHES) * (is_incrementing + 1)
        )
    assert num_samples_exported == num_samples_expected


def test_prediction_meter(tmp_path):
    y = [obj_det_y_i]
    y_pred = [obj_det_y_i_pred]
    y_pred_adv = [obj_det_y_i_pred]

    pred_meter = PredictionMeter(
        "pred_dict_exporter",
        tmp_path,
        y_probe="scenario.y",
        y_pred_clean_probe="scenario.y_pred",
        y_pred_adv_probe="scenario.y_pred_adv",
    )
    hub.connect_meter(pred_meter, use_default_writers=False)

    pred_meter_max_batch_1 = PredictionMeter(
        "pred_dict_exporter",
        tmp_path,
        y_probe="scenario.y",
        y_pred_clean_probe="scenario.y_pred",
        y_pred_adv_probe="scenario.y_pred_adv",
        max_batches=1,
    )
    hub.connect_meter(pred_meter_max_batch_1, use_default_writers=False)

    probe = get_probe("scenario")
    for i in range(NUM_BATCHES):
        hub.set_context(batch=i)
        probe.update(y=y, y_pred=y_pred, y_pred_adv=y_pred_adv)

        assert i in pred_meter.y_dict.keys()
        for key_value in ["y", "y_pred", "y_pred_adv"]:
            assert key_value in pred_meter.y_dict.get(i)

    assert pred_meter.examples_saved == NUM_BATCHES
    assert len(pred_meter.y_dict) == NUM_BATCHES

    pred_meter.finalize()
    assert os.path.isfile(f"{tmp_path}/predictions.pkl")

    assert 0 in pred_meter_max_batch_1.y_dict.keys()
    assert len(pred_meter_max_batch_1.y_dict) == 1


def test_coco_box_format_meter(tmp_path):
    y = [obj_det_y_i]
    y_pred = [obj_det_y_i_pred]
    y_pred_adv = [obj_det_y_i_pred]

    coco_box_format_meter = CocoBoxFormatMeter(
        "coco_box_format_meter",
        tmp_path,
        y_probe="scenario.y",
        y_pred_clean_probe="scenario.y_pred",
        y_pred_adv_probe="scenario.y_pred_adv",
    )
    hub.connect_meter(coco_box_format_meter, use_default_writers=False)

    coco_box_format_meter_max_batch_1 = CocoBoxFormatMeter(
        "coco_box_format_meter_max_batch_1",
        tmp_path,
        y_probe="scenario.y",
        y_pred_clean_probe="scenario.y_pred",
        y_pred_adv_probe="scenario.y_pred_adv",
        max_batches=1,
    )
    hub.connect_meter(coco_box_format_meter_max_batch_1, use_default_writers=False)

    probe = get_probe("scenario")
    for i in range(NUM_BATCHES):
        hub.set_context(batch=i)
        probe.update(y=y, y_pred=y_pred, y_pred_adv=y_pred_adv)

    for box_list in [
        coco_box_format_meter.y_boxes_coco_format,
        coco_box_format_meter.y_pred_clean_boxes_coco_format,
        coco_box_format_meter.y_pred_adv_boxes_coco_format,
    ]:
        assert isinstance(box_list, list)
        assert len(box_list) == NUM_BATCHES
        for coco_formatted_dict in box_list:
            assert isinstance(coco_formatted_dict, dict)
            for key_value in ["image_id", "category_id", "bbox"]:
                assert key_value in coco_formatted_dict

    for box_list in [
        coco_box_format_meter_max_batch_1.y_boxes_coco_format,
        coco_box_format_meter_max_batch_1.y_pred_clean_boxes_coco_format,
        coco_box_format_meter_max_batch_1.y_pred_adv_boxes_coco_format,
    ]:
        assert isinstance(box_list, list)
        assert len(box_list) == 1


@pytest.mark.docker_required
def test_ffmpeg_library():
    completed = subprocess.run(["ffmpeg", "-encoders"], capture_output=True)
    assert "libx264" in completed.stdout.decode("utf-8")


def test_carla_depth_format_conversion_utility_functions():
    x_lin = np.array([10, 50, 100, 500, 1000])
    r, g, b = linear_depth_to_rgb(x_lin)
    x_lin_ = rgb_depth_to_linear(r, g, b)
    assert np.allclose(x_lin, x_lin_)

    r_, g_, b_ = linear_depth_to_rgb(x_lin_)
    assert np.allclose(r, r_)
    assert np.allclose(g, g_)
    assert np.allclose(b, b_)

    x_log = linear_to_log(x_lin)
    x_lin_ = log_to_linear(x_log)
    assert np.allclose(x_lin, x_lin_)

    x_log_ = linear_to_log(x_lin_)
    assert np.allclose(x_log, x_log_)
