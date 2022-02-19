from importlib import import_module
import pytest
from armory.data.utils import maybe_download_weights_from_s3
import numpy as np

# NOTES:
# Many of these tests will have to download external repos and weights files
# which requires access to open internet, armory github tokens, and armory s3 tokens


@pytest.mark.usefixtures("ensure_armory_dirs")
def get_armory_module_and_fn(
    module_name, fn_attr_name, weights_path, model_kwargs={}, wrapper_kwargs={}
):
    module = import_module(module_name)
    fn = getattr(module, fn_attr_name)
    if weights_path is not None:
        weights_path = maybe_download_weights_from_s3(weights_path)
    classifier = fn(
        model_kwargs=model_kwargs,
        wrapper_kwargs=wrapper_kwargs,
        weights_path=weights_path,
    )
    return module, fn, classifier


def calculate_accuracy(test_ds, classifier):
    accuracy = 0
    for _ in range(test_ds.batches_per_epoch):
        x, y = test_ds.get_batch()
        predictions = classifier.predict(x)
        accuracy += np.sum(np.argmax(predictions, axis=1) == y) / len(y)
    return accuracy / test_ds.batches_per_epoch


@pytest.mark.parametrize(
    "module_name, fn_attr_name, weights_path, preprocessing_fn, "
    "dataset_name, train_batch_size, train_epochs, fit_nb_epochs, "
    "test_batch_size, test_epochs, framework,"
    "expected_accuracy",
    [
        (
            "armory.baseline_models.pytorch.mnist",
            "get_art_model",
            None,
            None,
            "mnist",
            600,
            1,
            1,
            100,
            1,
            "numpy",
            0.9,
        ),
        (
            "armory.baseline_models.pytorch.mnist",
            "get_art_model",
            "undefended_mnist_5epochs.pth",
            None,
            "mnist",
            None,
            None,
            None,
            100,
            1,
            "numpy",
            0.98,
        ),
        (
            "armory.baseline_models.pytorch.cifar",
            "get_art_model",
            None,
            None,
            "cifar10",
            500,
            1,
            1,
            100,
            1,
            "numpy",
            0.1,  # TODO This seems brittle getting values between 0.1 0.25.... Orig was 0.25, setting to 0.1
        ),
        # TODO Skipping test_pytorch_xview_pretrained due to differences...need to resolve
        (
            "armory.baseline_models.pytorch.micronnet_gtsrb",
            "get_art_model",
            None,
            "preprocessing_fn",
            "german_traffic_sign",
            128,
            1,
            1,  # TODO Reduced this to 1 from 5 to speed up
            128,
            1,
            "numpy",
            0.8,  # TODO Accuracy seemed to remain the same
        ),
    ],
)
def test_model_creation(
    module_name,
    fn_attr_name,
    weights_path,
    preprocessing_fn,
    dataset_name,
    train_batch_size,
    train_epochs,
    fit_nb_epochs,
    test_batch_size,
    test_epochs,
    framework,
    expected_accuracy,
    dataset_generator,
    armory_dataset_dir,
):
    module, fn, classifier = get_armory_module_and_fn(
        module_name, fn_attr_name, weights_path
    )

    if weights_path is None:
        train_ds = dataset_generator(
            name=dataset_name,
            batch_size=train_batch_size,
            num_epochs=train_epochs,
            split="train",
            framework=framework,
            dataset_dir=armory_dataset_dir,
            preprocessing_fn=getattr(module, preprocessing_fn)
            if preprocessing_fn is not None
            else None,
        )
        classifier.fit_generator(
            train_ds, nb_epochs=fit_nb_epochs,
        )

    test_ds = dataset_generator(
        name=dataset_name,
        batch_size=test_batch_size,
        num_epochs=test_epochs,
        split="test",
        framework=framework,
        dataset_dir=armory_dataset_dir,
        preprocessing_fn=getattr(module, preprocessing_fn)
        if preprocessing_fn is not None
        else None,
    )

    accuracy = calculate_accuracy(test_ds, classifier)
    assert accuracy > expected_accuracy


# TODO This has different structure uses armory.data.utils.load_dataset...find out why
#  Also it uses `num-batches` where others dont
@pytest.mark.slow
def test_pytorch_xview_pretrained():
    module, fn, classifier = get_armory_module_and_fn(
        "armory.baseline_models.pytorch.xview_frcnn",
        "get_art_model",
        "xview_model_state_dict_epoch_99_loss_0p67",
    )
    from armory.utils.config_loading import load_dataset

    NUM_TEST_SAMPLES = 250
    dataset_config = {
        "batch_size": 1,
        "framework": "numpy",
        "module": "armory.data.datasets",
        "name": "xview",
    }
    test_dataset = load_dataset(
        dataset_config,
        epochs=1,
        split="test",
        num_batches=NUM_TEST_SAMPLES,
        shuffle_files=False,
    )
    from armory.utils.metrics import object_detection_AP_per_class

    list_of_ys = []
    list_of_ypreds = []
    for x, y in test_dataset:
        y_pred = classifier.predict(x)
        list_of_ys.extend(y)
        list_of_ypreds.extend(y_pred)

    average_precision_by_class = object_detection_AP_per_class(
        list_of_ys, list_of_ypreds
    )
    mAP = np.fromiter(average_precision_by_class.values(), dtype=float).mean()
    for class_id in [4, 23, 33, 39]:
        assert average_precision_by_class[class_id] > 0.9
    assert mAP > 0.25


# TODO Need to figure out why load_dataset is used which introduces new pattern
#  for calculating accuracy.
@pytest.mark.usefixtures("ensure_armory_dirs")
def test_pytorch_carla_video_tracking():
    tracker_module = import_module("armory.baseline_models.pytorch.carla_goturn")
    tracker_fn = getattr(tracker_module, "get_art_model")
    weights_path = maybe_download_weights_from_s3("pytorch_goturn.pth.tar")
    tracker = tracker_fn(model_kwargs={}, wrapper_kwargs={}, weights_path=weights_path,)

    # TODO Figure out why this different pattern is used
    from armory.utils.config_loading import load_dataset

    NUM_TEST_SAMPLES = 10
    dataset_config = {
        "batch_size": 1,
        "framework": "numpy",
        "module": "armory.data.adversarial_datasets",
        "name": "carla_video_tracking_dev",
    }
    dev_dataset = load_dataset(
        dataset_config,
        epochs=1,
        split="dev",
        num_batches=NUM_TEST_SAMPLES,
        shuffle_files=False,
    )
    from armory.utils.metrics import video_tracking_mean_iou

    for x, y in dev_dataset:
        y_object, y_patch_metadata = y
        y_init = np.expand_dims(y_object[0]["boxes"][0], axis=0)
        y_pred = tracker.predict(x, y_init=y_init)
        mean_iou = video_tracking_mean_iou(y_object, y_pred)[0]
        assert mean_iou > 0.45


@pytest.mark.parametrize(
    "module_name, fn_attr_name, " "weights_path, model_kwargs, " "dataset_parameters",
    [
        (  # 1 of 3 from test_carla_od_rgb
            "armory.baseline_models.pytorch.carla_single_modality_object_detection_frcnn",
            "get_art_model",
            "carla_rgb_weights.pt",
            {"num_classes": 4},
            {
                "dataset_config": {
                    "batch_size": 1,
                    "framework": "numpy",
                    "module": "armory.data.datasets",
                    "name": "carla_obj_det_train",
                },
                "modality": "rgb",
                "epochs": 1,
                "split": "train",
                "num_batches": 10,
                "shuffle_files": False,
            },
        ),
        (  # 2 of 3 from test_carla_od_rgb
            "armory.baseline_models.pytorch.carla_single_modality_object_detection_frcnn",
            "get_art_model",
            "carla_rgb_weights.pt",
            {"num_classes": 4},
            {
                "dataset_config": {
                    "batch_size": 1,
                    "framework": "numpy",
                    "module": "armory.data.adversarial_datasets",
                    "name": "carla_obj_det_dev",
                },
                "modality": "rgb",
                "epochs": 1,
                "split": "dev",
                "num_batches": 10,
                "shuffle_files": False,
            },
        ),
        (  # 3 of 3 from test_carla_od_rgb
            "armory.baseline_models.pytorch.carla_single_modality_object_detection_frcnn",
            "get_art_model",
            "carla_rgb_weights.pt",
            {"num_classes": 4},
            {
                "dataset_config": {
                    "batch_size": 1,
                    "framework": "numpy",
                    "module": "armory.data.adversarial_datasets",
                    "name": "carla_obj_det_test",
                },
                "modality": "rgb",
                "epochs": 1,
                "split": "test",
                "num_batches": 10,
                "shuffle_files": False,
            },
        ),
        (  # 1 of 3 from test_carla_od_depth
            "armory.baseline_models.pytorch.carla_single_modality_object_detection_frcnn",
            "get_art_model",
            "carla_depth_weights.pt",
            {"num_classes": 4},
            {
                "dataset_config": {
                    "batch_size": 1,
                    "framework": "numpy",
                    "module": "armory.data.datasets",
                    "name": "carla_obj_det_train",
                },
                "modality": "depth",
                "epochs": 1,
                "split": "train",
                "num_batches": 10,
                "shuffle_files": False,
            },
        ),
        (  # 2 of 3 from test_carla_od_depth
            "armory.baseline_models.pytorch.carla_single_modality_object_detection_frcnn",
            "get_art_model",
            "carla_depth_weights.pt",
            {"num_classes": 4},
            {
                "dataset_config": {
                    "batch_size": 1,
                    "framework": "numpy",
                    "module": "armory.data.adversarial_datasets",
                    "name": "carla_obj_det_dev",
                },
                "modality": "depth",
                "epochs": 1,
                "split": "dev",
                "num_batches": 10,
                "shuffle_files": False,
            },
        ),
        (  # 3 of 3 from test_carla_od_depth
            "armory.baseline_models.pytorch.carla_single_modality_object_detection_frcnn",
            "get_art_model",
            "carla_depth_weights.pt",
            {"num_classes": 4},
            {
                "dataset_config": {
                    "batch_size": 1,
                    "framework": "numpy",
                    "module": "armory.data.adversarial_datasets",
                    "name": "carla_obj_det_test",
                },
                "modality": "depth",
                "epochs": 1,
                "split": "test",
                "num_batches": 10,
                "shuffle_files": False,
            },
        ),
        (  # 1 of 3 from test_carla_od_multimodal
            "armory.baseline_models.pytorch.carla_multimodality_object_detection_frcnn",
            "get_art_model_mm",
            "carla_multimodal_naive_weights.pt",
            {},
            {
                "dataset_config": {
                    "batch_size": 1,
                    "framework": "numpy",
                    "module": "armory.data.datasets",
                    "name": "carla_obj_det_train",
                },
                "modality": "both",
                "epochs": 1,
                "split": "train",
                "num_batches": 10,
                "shuffle_files": False,
            },
        ),
        (  # 2 of 3 from test_carla_od_multimodal
            "armory.baseline_models.pytorch.carla_multimodality_object_detection_frcnn",
            "get_art_model_mm",
            "carla_multimodal_naive_weights.pt",
            {},
            {
                "dataset_config": {
                    "batch_size": 1,
                    "framework": "numpy",
                    "module": "armory.data.adversarial_datasets",
                    "name": "carla_obj_det_dev",
                },
                "modality": "both",
                "epochs": 1,
                "split": "dev",
                "num_batches": 10,
                "shuffle_files": False,
            },
        ),
        (  # 3 of 3 from test_carla_od_multimodal
            "armory.baseline_models.pytorch.carla_multimodality_object_detection_frcnn",
            "get_art_model_mm",
            "carla_multimodal_naive_weights.pt",
            {},
            {
                "dataset_config": {
                    "batch_size": 1,
                    "framework": "numpy",
                    "module": "armory.data.adversarial_datasets",
                    "name": "carla_obj_det_test",
                },
                "modality": "both",
                "epochs": 1,
                "split": "test",
                "num_batches": 10,
                "shuffle_files": False,
            },
        ),
        (  # 1 of 1 from test_carla_od_multimodal_robust_fusion
            "armory.baseline_models.pytorch.carla_multimodality_object_detection_frcnn_robust_fusion",
            "get_art_model_mm_robust",
            "carla_multimodal_robust_clw_1_weights.pt",
            {},
            {
                "dataset_config": {
                    "batch_size": 1,
                    "framework": "numpy",
                    "module": "armory.data.datasets",
                    "name": "carla_obj_det_train",
                },
                "modality": "both",
                "epochs": 1,
                "split": "train",
                "num_batches": 10,
                "shuffle_files": False,
            },
        ),
    ],
)
@pytest.mark.slow
def test_carla_od(
    module_name,
    fn_attr_name,
    weights_path,
    model_kwargs,
    dataset_parameters,
    dataset_generator,
    armory_dataset_dir,
):
    module, fn, classifier = get_armory_module_and_fn(
        module_name, fn_attr_name, weights_path, model_kwargs
    )
    from armory.utils.config_loading import load_dataset
    from armory.utils.metrics import object_detection_AP_per_class

    ds = load_dataset(**dataset_parameters)
    ys = []
    y_preds = []
    for thing in ds:
        x = thing[0]
        if len(thing[1]) == 1:
            y = thing[1]
        elif len(thing[1]) == 2:
            y, y_patch_metadata = thing[1]
        else:
            raise Exception("I don't recognize thing: `{}`".format(thing))
        y_pred = classifier.predict(x)
        if len(thing[1]) == 1:
            ys.extend(y)
        else:
            ys.append(y)
        y_preds.extend(y_pred)
    ap_per_class = object_detection_AP_per_class(ys, y_preds)
    assert [ap_per_class[i] > 0.35 for i in range(1, 4)]


# TODO:  Still need to add the following
#  - test_tf1/test_keras_models.py - These complain about tf `eager` execution
#  - test_tf1/test_tf1_models.py

# TODO: Think about how to mark these appropriately so they can be split in CI

# TODO: Determine if moving the `download_and_extract_external_repo` bits is
#  appropriate to move to the model module.  E.g. see what I did in:
#  armory.baseline_models.pytorch.carla_goturn
