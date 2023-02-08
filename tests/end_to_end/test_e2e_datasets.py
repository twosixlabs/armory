import os

import numpy as np
import pytest
import tensorflow as tf
import torch

from armory.data import adversarial_datasets, datasets

# Marks all tests in this file as `end_to_end`
pytestmark = pytest.mark.end_to_end


@pytest.mark.parametrize(
    "name, batch_size, num_epochs, split, framework, xtype, xshape, ytype, yshape, dataset_size",
    [
        (
            "cifar10",
            16,
            1,
            "train",
            "pytorch",
            torch.uint8,
            (16, 32, 32, 3),
            torch.int64,
            (16,),
            50000,
        ),
        (
            "mnist",
            16,
            1,
            "train",
            "pytorch",
            torch.uint8,
            (16, 28, 28, 1),
            torch.int64,
            (16,),
            60000,
        ),
        (
            "resisc45",
            16,
            1,
            "train",
            "pytorch",
            torch.uint8,
            (16, 256, 256, 3),
            torch.int64,
            (16,),
            22500,
        ),
        (
            "mnist",
            16,
            1,
            "train",
            "tf",
            tf.uint8,
            (16, 28, 28, 1),
            tf.int64,
            (16,),
            60000,
        ),
        (
            "mnist",
            16,
            1,
            "train",
            "numpy",
            tf.uint8,
            (16, 28, 28, 1),
            tf.int64,
            (16,),
            60000,
        ),
        # These come from the tests/test_docker/test_dataset.py individuals
        (
            "mnist",
            600,
            1,
            "train",
            "numpy",
            tf.uint8,
            (600, 28, 28, 1),
            tf.int64,
            (600,),
            60000,
        ),
        (
            "mnist",
            600,
            1,
            "test",
            "numpy",
            tf.uint8,
            (600, 28, 28, 1),
            tf.int64,
            (600,),
            10000,
        ),
        (
            "cifar10",
            500,
            1,
            "train",
            "numpy",
            tf.uint8,
            (500, 32, 32, 3),
            tf.int64,
            (500,),
            50000,
        ),
        (
            "cifar10",
            500,
            1,
            "test",
            "numpy",
            tf.uint8,
            (500, 32, 32, 3),
            tf.int64,
            (500,),
            10000,
        ),
        (
            "cifar100",
            500,
            1,
            "train",
            "numpy",
            tf.uint8,
            (500, 32, 32, 3),
            tf.int64,
            (500,),
            50000,
        ),
        (
            "cifar100",
            500,
            1,
            "test",
            "numpy",
            tf.uint8,
            (500, 32, 32, 3),
            tf.int64,
            (500,),
            10000,
        ),
        (
            "resisc45",
            16,
            1,
            "train",
            "numpy",
            tf.uint8,
            (16, 256, 256, 3),
            tf.int64,
            (16,),
            22500,
        ),
        (
            "resisc45",
            16,
            1,
            "test",
            "numpy",
            tf.uint8,
            (16, 256, 256, 3),
            tf.int64,
            (16,),
            4500,
        ),
        (
            "resisc45",
            16,
            1,
            "validation",
            "numpy",
            tf.uint8,
            (16, 256, 256, 3),
            tf.int64,
            (16,),
            4500,
        ),
        (
            "resisc10",
            16,
            1,
            "train",
            "numpy",
            tf.uint8,
            (16, 256, 256, 3),
            tf.int64,
            (16,),
            5000,
        ),
        (
            "resisc10",
            16,
            1,
            "test",
            "numpy",
            tf.uint8,
            (16, 256, 256, 3),
            tf.int64,
            (16,),
            1000,
        ),
        (
            "resisc10",
            16,
            1,
            "validation",
            "numpy",
            tf.uint8,
            (16, 256, 256, 3),
            tf.int64,
            (16,),
            1000,
        ),
    ],
)
def test_generator_construction(
    name,
    batch_size,
    num_epochs,
    split,
    framework,
    xtype,
    xshape,
    ytype,
    yshape,
    dataset_size,
    dataset_generator,
    armory_dataset_dir,
):
    dataset = dataset_generator(
        name, batch_size, num_epochs, split, framework, dataset_dir=armory_dataset_dir
    )

    x, y = next(iter(dataset))

    assert x.dtype == xtype
    assert x.shape == xshape
    assert y.dtype == ytype
    assert y.shape == yshape
    if framework == "numpy":
        assert dataset.size == dataset_size
        x, y = dataset.get_batch()
        assert x.shape == xshape
        assert y.shape == yshape
        assert isinstance(x, np.ndarray)


# TODO Clean up tests below here


# TODO Talk to David to see why these parameters are used....could we just used the
#  bits from the tests_datasets.py::test_generator_construction
#  David says these can be removed once we do all the asserts


def test_digit(armory_dataset_dir):

    epochs = 1
    batch_size = 1
    num_users = 3
    min_length = 1148
    max_length = 18262
    for split, size in [
        ("train", 45 * num_users * 10),
        ("test", 5 * num_users * 10),
    ]:
        dataset = datasets.digit(
            split=split,
            epochs=epochs,
            batch_size=batch_size,
            dataset_dir=armory_dataset_dir,
        )
        assert dataset.size == size
        assert dataset.batch_size == batch_size

        x, y = dataset.get_batch()
        assert x.shape[0] == batch_size
        assert x.ndim == 2
        assert min_length <= x.shape[1] <= max_length
        assert y.shape == (batch_size,)


def test_imagenet_adv(armory_dataset_dir):

    batch_size = 100
    total_size = 1000
    test_dataset = adversarial_datasets.imagenet_adversarial(
        dataset_dir=armory_dataset_dir,
        split="adversarial",
        batch_size=batch_size,
        epochs=1,
        adversarial_key="adversarial",
    )
    assert test_dataset.size == total_size
    assert test_dataset.batch_size == batch_size
    assert test_dataset.batches_per_epoch == (
        total_size // batch_size + bool(total_size % batch_size)
    )

    x, y = test_dataset.get_batch()
    for i in range(2):
        assert x[i].shape == (batch_size, 224, 224, 3)
    assert y.shape == (batch_size,)


def test_german_traffic_sign(armory_dataset_dir):

    for split, size in [("train", 39209), ("test", 12630)]:
        batch_size = 1
        epochs = 1
        dataset = datasets.german_traffic_sign(
            split=split,
            epochs=epochs,
            batch_size=batch_size,
            dataset_dir=armory_dataset_dir,
        )
        assert dataset.size == size

        x, y = dataset.get_batch()
        # sign image shape is variable so we don't compare 2nd dim
        assert x.shape[:1] + x.shape[3:] == (batch_size, 3)
        assert y.shape == (batch_size,)


def test_imagenette(armory_dataset_dir):

    if not os.path.isdir(
        os.path.join(armory_dataset_dir, "imagenette", "full-size", "0.1.0")
    ):
        pytest.skip("imagenette dataset not locally available.")

    for split, size in [("train", 12894), ("validation", 500)]:
        batch_size = 1
        epochs = 1
        dataset = datasets.imagenette(
            split=split,
            epochs=epochs,
            batch_size=batch_size,
            dataset_dir=armory_dataset_dir,
        )
        assert dataset.size == size

        x, y = dataset.get_batch()
        # image dimensions are variable so we don't compare 2nd dim or 3rd dim
        assert x.shape[:1] + x.shape[3:] == (batch_size, 3)
        assert y.shape == (batch_size,)


def test_ucf101(armory_dataset_dir):

    if not os.path.isdir(
        os.path.join(armory_dataset_dir, "ucf101", "ucf101_1", "2.0.0")
    ):
        pytest.skip("ucf101 dataset not locally available.")

    for split, size in [("train", 9537), ("test", 3783)]:
        batch_size = 1
        epochs = 1
        dataset = datasets.ucf101(
            split=split,
            epochs=epochs,
            batch_size=batch_size,
            dataset_dir=armory_dataset_dir,
        )
        assert dataset.size == size

        x, y = dataset.get_batch()
        # video length is variable so we don't compare 2nd dim
        assert x.shape[:1] + x.shape[2:] == (batch_size, 240, 320, 3)
        assert y.shape == (batch_size,)


def test_librispeech(armory_dataset_dir):

    if not os.path.exists(
        os.path.join(armory_dataset_dir, "librispeech_dev_clean_split")
    ):
        pytest.skip("Librispeech dataset not downloaded.")

    splits = ("train", "validation", "test")
    sizes = (1371, 692, 640)
    min_dim1s = (23120, 26239, 24080)
    max_dim1s = (519760, 516960, 522320)
    batch_size = 1

    for split, size, min_dim1, max_dim1 in zip(splits, sizes, min_dim1s, max_dim1s):
        dataset = datasets.librispeech_dev_clean(
            split=split,
            epochs=1,
            batch_size=batch_size,
            dataset_dir=armory_dataset_dir,
        )
        assert dataset.size == size
        assert dataset.batch_size == batch_size
        assert dataset.batches_per_epoch == (
            size // batch_size + bool(size % batch_size)
        )

        x, y = dataset.get_batch()
        assert x.shape[0] == 1
        assert min_dim1 <= x.shape[1] <= max_dim1
        assert y.shape == (batch_size,)


def test_librispeech_adversarial(armory_dataset_dir):

    if not os.path.exists(
        os.path.join(armory_dataset_dir, "librispeech_adversarial", "1.0.0")
    ):
        pytest.skip("Librispeech adversarial dataset not downloaded.")

    size = 2703
    min_dim1 = 23120
    max_dim1 = 522320
    batch_size = 1
    split = "adversarial"

    dataset = adversarial_datasets.librispeech_adversarial(
        split=split,
        epochs=1,
        batch_size=batch_size,
        dataset_dir=armory_dataset_dir,
        adversarial_key="adversarial",
    )
    assert dataset.size == size
    assert dataset.batch_size == batch_size
    assert dataset.batches_per_epoch == (size // batch_size + bool(size % batch_size))

    x, y = dataset.get_batch()
    for i in range(2):
        assert x[i].shape[0] == 1
        assert min_dim1 <= x[i].shape[1] <= max_dim1
    assert y.shape == (batch_size,)


def test_resisc45_adversarial_224x224(armory_dataset_dir):

    size = 225
    split = "adversarial"
    batch_size = 16
    epochs = 1
    for adversarial_key in ("adversarial_univpatch", "adversarial_univperturbation"):
        dataset = adversarial_datasets.resisc45_adversarial_224x224(
            split=split,
            epochs=epochs,
            batch_size=batch_size,
            dataset_dir=armory_dataset_dir,
            adversarial_key=adversarial_key,
        )
        assert dataset.size == size
        assert dataset.batch_size == batch_size
        assert dataset.batches_per_epoch == (
            size // batch_size + bool(size % batch_size)
        )

        x, y = dataset.get_batch()
        for i in range(2):
            assert x[i].shape == (batch_size, 224, 224, 3)
        assert y.shape == (batch_size,)


def test_coco2017(armory_dataset_dir):

    if not os.path.exists(os.path.join(armory_dataset_dir, "coco", "2017", "1.1.0")):
        pytest.skip("coco2017 dataset not downloaded.")

    split_size = 5000
    split = "validation"
    dataset = datasets.coco2017(
        split=split,
    )
    assert dataset.size == split_size

    for i in range(8):
        x, y = dataset.get_batch()
        assert x.shape[0] == 1
        assert x.shape[-1] == 3
        assert isinstance(y, list)
        assert len(y) == 1
        y_dict = y[0]
        assert isinstance(y_dict, dict)
        for obj_key in ["labels", "boxes", "area"]:
            assert obj_key in y_dict


def test_dapricot_dev():

    split_size = 27
    split = "small"
    dataset = adversarial_datasets.dapricot_dev_adversarial(
        split=split,
    )
    assert dataset.size == split_size

    x, y = dataset.get_batch()
    for i in range(2):
        assert x.shape == (1, 3, 1008, 756, 3)
        assert isinstance(y, tuple)
        assert len(y) == 2
        y_object, y_patch_metadata = y
        assert len(y_object) == 3  # 3 images per example
        for obj_key in ["labels", "boxes", "area"]:
            for k in range(3):
                assert obj_key in y_object[k]
        for patch_metadata_key in ["cc_scene", "cc_ground_truth", "gs_coords", "shape"]:
            for k in range(3):
                assert patch_metadata_key in y_patch_metadata[k]


def test_dapricot_test():

    split_size = 108
    split = "small"
    dataset = adversarial_datasets.dapricot_test_adversarial(
        split=split,
    )
    assert dataset.size == split_size

    x, y = dataset.get_batch()
    for i in range(2):
        assert x.shape == (1, 3, 1008, 756, 3)
        assert isinstance(y, tuple)
        assert len(y) == 2
        y_object, y_patch_metadata = y
        assert len(y_object) == 3  # 3 images per example
        for obj_key in ["labels", "boxes", "area"]:
            for k in range(3):
                assert obj_key in y_object[k]
        for patch_metadata_key in ["cc_scene", "cc_ground_truth", "gs_coords", "shape"]:
            for k in range(3):
                assert patch_metadata_key in y_patch_metadata[k]


def test_carla_obj_det_train():
    # Testing batch_size > 1
    batch_size = 2

    for split in ["train", "val"]:
        for modality in ["rgb", "depth", "both"]:
            expected_shape = (
                (batch_size, 960, 1280, 6)
                if modality == "both"
                else (batch_size, 960, 1280, 3)
            )
            ds_batch_size2 = datasets.carla_obj_det_train(
                split=split, batch_size=batch_size, modality=modality
            )
            if split == "train":
                assert ds_batch_size2.size == 3496
            elif split == "val":
                assert ds_batch_size2.size == 1200

            x, y = ds_batch_size2.get_batch()
            assert x.shape == expected_shape
            assert len(y) == batch_size
            for label_dict in y:
                assert isinstance(label_dict, dict)
                for obj_key in ["labels", "boxes", "area"]:
                    assert obj_key in label_dict


def test_carla_obj_det_dev():

    ds_rgb = adversarial_datasets.carla_obj_det_dev(split="dev", modality="rgb")
    ds_depth = adversarial_datasets.carla_obj_det_dev(split="dev", modality="depth")
    ds_multimodal = adversarial_datasets.carla_obj_det_dev(split="dev", modality="both")

    for i, ds in enumerate([ds_multimodal, ds_rgb, ds_depth]):
        for x, y in ds:
            if i == 0:
                assert x.shape == (1, 960, 1280, 6)
            else:
                assert x.shape == (1, 960, 1280, 3)

            y_object, y_patch_metadata = y
            assert isinstance(y_object, dict)
            for obj_key in ["labels", "boxes", "area"]:
                assert obj_key in y_object
            assert isinstance(y_patch_metadata, dict)
            for patch_key in [
                "cc_ground_truth",
                "cc_scene",
                "gs_coords",
                "mask",
                "shape",
            ]:
                assert patch_key in y_patch_metadata

    with pytest.raises(ValueError):
        ds = adversarial_datasets.carla_obj_det_dev(
            split="dev", modality="invalid_string"
        )


def test_carla_obj_det_test():

    ds_rgb = adversarial_datasets.carla_obj_det_test(split="test", modality="rgb")
    ds_depth = adversarial_datasets.carla_obj_det_test(split="test", modality="depth")
    ds_multimodal = adversarial_datasets.carla_obj_det_test(
        split="test", modality="both"
    )
    for i, ds in enumerate([ds_multimodal, ds_rgb, ds_depth]):
        for x, y in ds:
            if i == 0:
                assert x.shape == (1, 960, 1280, 6)
            else:
                assert x.shape == (1, 960, 1280, 3)

            y_object, y_patch_metadata = y
            assert isinstance(y_object, dict)
            for obj_key in ["labels", "boxes", "area"]:
                assert obj_key in y_object
            assert isinstance(y_patch_metadata, dict)
            for patch_key in [
                "cc_ground_truth",
                "cc_scene",
                "gs_coords",
                "mask",
                "shape",
            ]:
                assert patch_key in y_patch_metadata

    with pytest.raises(ValueError):
        ds = adversarial_datasets.carla_obj_det_dev(
            split="dev", modality="invalid_string"
        )


def test_carla_overhead_obj_det_train():
    # Testing batch_size > 1
    batch_size = 2

    for split in ["train", "val"]:
        for modality in ["rgb", "depth", "both"]:
            expected_shape = (
                (batch_size, 960, 1280, 6)
                if modality == "both"
                else (batch_size, 960, 1280, 3)
            )
            ds_batch_size2 = datasets.carla_over_obj_det_train(
                split=split, batch_size=batch_size, modality=modality
            )
            if split == "train":
                assert ds_batch_size2.size == 3600
            elif split == "val":
                assert ds_batch_size2.size == 1200

            x, y = ds_batch_size2.get_batch()
            assert x.shape == expected_shape
            assert len(y) == batch_size
            for label_dict in y:
                assert isinstance(label_dict, dict)
                for obj_key in [
                    "area",
                    "boxes",
                    "id",
                    "image_id",
                    "is_crowd",
                    "labels",
                ]:
                    assert obj_key in label_dict


def test_carla_overhead_obj_det_dev():

    ds_rgb = adversarial_datasets.carla_over_obj_det_dev(split="dev", modality="rgb")
    ds_depth = adversarial_datasets.carla_over_obj_det_dev(
        split="dev", modality="depth"
    )
    ds_multimodal = adversarial_datasets.carla_over_obj_det_dev(
        split="dev", modality="both"
    )

    for i, ds in enumerate([ds_multimodal, ds_rgb, ds_depth]):
        for x, y in ds:
            if i == 0:
                assert x.shape == (1, 960, 1280, 6)
            else:
                assert x.shape == (1, 960, 1280, 3)

            y_object, y_patch_metadata = y
            assert isinstance(y_object, dict)
            for obj_key in ["labels", "boxes", "area"]:
                assert obj_key in y_object
            assert isinstance(y_patch_metadata, dict)
            for patch_key in [
                "avg_patch_depth",
                "gs_coords",
                "mask",
            ]:
                assert patch_key in y_patch_metadata

    with pytest.raises(ValueError):
        ds = adversarial_datasets.carla_over_obj_det_dev(
            split="dev", modality="invalid_string"
        )


def test_carla_overhead_obj_det_test():

    ds_rgb = adversarial_datasets.carla_over_obj_det_test(split="test", modality="rgb")
    ds_depth = adversarial_datasets.carla_over_obj_det_test(
        split="test", modality="depth"
    )
    ds_multimodal = adversarial_datasets.carla_over_obj_det_test(
        split="test", modality="both"
    )

    for i, ds in enumerate([ds_multimodal, ds_rgb, ds_depth]):
        assert ds.size == 15
        for x, y in ds:
            if i == 0:
                assert x.shape == (1, 960, 1280, 6)
            else:
                assert x.shape == (1, 960, 1280, 3)

            y_object, y_patch_metadata = y
            assert isinstance(y_object, dict)
            for obj_key in ["labels", "boxes", "area"]:
                assert obj_key in y_object
            assert isinstance(y_patch_metadata, dict)
            for patch_key in [
                "avg_patch_depth",
                "gs_coords",
                "mask",
            ]:
                assert patch_key in y_patch_metadata


def test_carla_video_tracking_dev():

    dataset = adversarial_datasets.carla_video_tracking_dev(split="dev")
    assert dataset.size == 20
    for x, y in dataset:
        assert x.shape[0] == 1
        assert x.shape[2:] == (960, 1280, 3)
        assert isinstance(y, tuple)
        assert len(y) == 2
        y_object, y_patch_metadata = y
        assert isinstance(y_object, list)
        assert len(y_object) == 1
        assert isinstance(y_object[0], dict)
        assert "boxes" in y_object[0]
        assert y_object[0]["boxes"].shape[1] == 4
        assert isinstance(y_patch_metadata, dict)
        for key in ["cc_ground_truth", "cc_scene", "gs_coords", "masks"]:
            assert key in y_patch_metadata


def test_carla_video_tracking_test():

    dataset = adversarial_datasets.carla_video_tracking_test(split="test")
    assert dataset.size == 20
    for x, y in dataset:
        assert x.shape[0] == 1
        assert x.shape[2:] == (960, 1280, 3)
        assert isinstance(y, tuple)
        assert len(y) == 2
        y_object, y_patch_metadata = y
        assert isinstance(y_object, list)
        assert len(y_object) == 1
        assert isinstance(y_object[0], dict)
        assert "boxes" in y_object[0]
        assert y_object[0]["boxes"].shape[1] == 4
        assert isinstance(y_patch_metadata, dict)
        for key in ["cc_ground_truth", "cc_scene", "gs_coords", "masks"]:
            assert key in y_patch_metadata


def test_ucf101_adversarial_112x112(armory_dataset_dir):

    if not os.path.isdir(
        os.path.join(
            armory_dataset_dir,
            "ucf101_mars_perturbation_and_patch_adversarial112x112",
            "1.0.0",
        )
    ):
        pytest.skip("ucf101 adversarial dataset not locally available.")

    for adversarial_key in ("adversarial_perturbation", "adversarial_patch"):
        batch_size = 1
        epochs = 1
        size = 505
        split = "adversarial"
        dataset = adversarial_datasets.ucf101_adversarial_112x112(
            split=split,
            epochs=epochs,
            batch_size=batch_size,
            dataset_dir=armory_dataset_dir,
            adversarial_key=adversarial_key,
        )
        assert dataset.size == size

        x, y = dataset.get_batch()
        for i in range(2):
            # video length is variable so we don't compare 2nd dim
            assert x[i].shape[:1] + x[i].shape[2:] == (batch_size, 112, 112, 3)
        assert y.shape == (batch_size,)


def test_speech_commands(armory_dataset_dir):

    batch_size = 16
    ds_test_size = 4890
    ds_train_size = 85511
    sample_length = 16000

    ds = datasets.speech_commands(
        split="test",
        pad_data=True,
        batch_size=batch_size,
        dataset_dir=armory_dataset_dir,
    )
    assert ds.size == ds_test_size
    assert ds.batch_size == batch_size

    x, y = ds.get_batch()
    assert x.shape[0] == batch_size
    assert x.shape[1] == sample_length
    assert y.shape == (batch_size,)

    ds = datasets.speech_commands(
        split="train",
        pad_data=True,
        batch_size=batch_size,
        dataset_dir=armory_dataset_dir,
    )
    assert ds.size == ds_train_size


def test_carla_multi_object_tracking_dev():

    dataset = adversarial_datasets.carla_multi_object_tracking_dev(split="dev")
    assert dataset.size == 20
    for x, y in dataset:
        assert x.shape[0] == 1
        assert x.shape[2:] == (960, 1280, 3)
        assert isinstance(y, tuple)
        assert len(y) == 2
        annotations, y_patch_metadata = y
        assert isinstance(annotations, np.ndarray)
        assert annotations.shape[0] == 1
        assert annotations.shape[2] == 9
        assert isinstance(y_patch_metadata, dict)
        for key in ["gs_coords", "masks"]:
            assert key in y_patch_metadata


def test_carla_multi_object_tracking_test():

    dataset = adversarial_datasets.carla_multi_object_tracking_test(split="test")
    assert dataset.size == 10
    for x, y in dataset:
        assert x.shape[0] == 1
        assert x.shape[2:] == (960, 1280, 3)
        assert isinstance(y, tuple)
        assert len(y) == 2
        annotations, y_patch_metadata = y
        assert isinstance(annotations, np.ndarray)
        assert annotations.shape[0] == 1
        assert annotations.shape[2] == 9
        assert isinstance(y_patch_metadata, dict)
        for key in ["gs_coords", "masks"]:
            assert key in y_patch_metadata


def test_variable_length(armory_dataset_dir):
    """
    Test batches with variable length items using digit dataset
    """
    size = 1350
    batch_size = 4

    dataset = datasets.digit(
        split="train",
        epochs=1,
        batch_size=batch_size,
        dataset_dir=armory_dataset_dir,
    )
    assert dataset.batches_per_epoch == (size // batch_size + bool(size % batch_size))

    x, y = dataset.get_batch()
    assert x.dtype == object
    assert x.shape == (batch_size,)
    for x_i in x:
        assert x_i.ndim == 1
        assert 1148 <= len(x_i) <= 18262
    assert y.shape == (batch_size,)
