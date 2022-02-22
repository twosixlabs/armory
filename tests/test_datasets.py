import torch
import numpy as np
import pytest
import logging
import tensorflow as tf
import os

log = logging.getLogger(__name__)


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

    # TODO rename to x, y
    x, y = next(iter(dataset))

    # TODO:  These asserts will work with numpy framework because its ArmoryDataGenerator
    #  If you select tf or pytorch it will give you the tf dataset apis, if you select
    #  pytorch it will send back dataloader with assert statements
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


@pytest.mark.parametrize(
    "name, batch_size, num_epochs, split, framework, shuffle, exp_cnt",
    [("mnist", 10, 2, "test", "pytorch", True, 2000,),],
)
def test_generator_epoch_creation(
    name,
    batch_size,
    num_epochs,
    split,
    framework,
    shuffle,
    exp_cnt,
    dataset_generator,
    armory_dataset_dir,
):
    dataset = dataset_generator(
        name=name,
        batch_size=batch_size,
        num_epochs=num_epochs,
        split=split,
        framework=framework,
        dataset_dir=armory_dataset_dir,
        shuffle_files=shuffle,
    )
    cnt = 0

    for images, labels in dataset:
        if cnt == 0:
            first_batch = labels
        if cnt == 1000:
            second_batch = labels
        cnt += 1

    assert cnt == exp_cnt

    if shuffle:
        assert not torch.all(torch.eq(first_batch, second_batch))
    else:
        assert torch.all(torch.eq(first_batch, second_batch))


@pytest.mark.parametrize(
    "name, batch_size, framework1, framework2", [("mnist", 10, "tf", "pytorch")]
)
def test_framework_equality(
    name, batch_size, framework1, framework2, dataset_generator, armory_dataset_dir
):
    ds1 = dataset_generator(
        name, batch_size, 1, "test", framework1, dataset_dir=armory_dataset_dir
    )
    ds2 = dataset_generator(
        name, batch_size, 1, "test", framework2, dataset_dir=armory_dataset_dir
    )

    for ex1, ex2 in zip(ds1, ds2):
        img1, l1 = ex1[0].numpy(), ex1[1].numpy()
        img2, l2 = ex2[0].numpy(), ex2[1].numpy()

        assert np.amax(np.abs(img1 - img2)) == 0
        assert np.amax(np.abs(l1 - l2)) == 0


# TODO: Check if there is a reason not to have this here as I moved it from
#  tests/test_docker/test_supported_datasets.py
def test_supported_datasets():
    from armory.data import datasets

    for name, function in datasets.SUPPORTED_DATASETS.items():
        assert callable(function)
        # TODO:  This seems super minimal....maybe consider invoking the callable here
        #  or add this as a part of the `test_generator_construction` test above.


# TODO:  The Following were moved from test_docker/test_dataset.py because the don't have
# anything to do with docker


@pytest.mark.parametrize(
    "dataset_method, is_valid, input1, input2",
    [
        ("_parse_token", True, "test", "test"),
        ("_parse_token", True, "train[15:20]", "train[15:20]"),
        ("_parse_token", True, "train[:10%]", "train[:10%]"),
        ("_parse_token", True, "train[-80%:]", "train[-80%:]"),
        ("_parse_token", True, "test[[1, 5, 7]]", "test[1:2]+test[5:6]+test[7:8]"),
        (
            "_parse_token",
            True,
            "test[[1, 4, 5, 6]]",
            "test[1:2]+test[4:5]+test[5:6]+test[6:7]",
        ),
        ("_parse_token", True, "test[10]", "test[10:11]"),
        ("_parse_token", False, "", ValueError),
        ("_parse_token", False, "test[", ValueError),
        ("_parse_token", False, "test[]", ValueError),
        ("_parse_token", False, "test[[]]", ValueError),
        ("_parse_token", False, "[10:11]", ValueError),
        ("_parse_token", False, "test[10:20:2]", NotImplementedError),
        ("parse_split_index", True, "train[15:20]", "train[15:20]"),
        (
            "parse_split_index",
            True,
            "train[:10%]+train[-80%:]",
            "train[:10%]+train[-80%:]",
        ),
        ("parse_split_index", True, "test[[1, 5, 7]]", "test[1:2]+test[5:6]+test[7:8]"),
        (
            "parse_split_index",
            True,
            "test[[1, 4, 5, 6]]",
            "test[1:2]+test[4:5]+test[5:6]+test[6:7]",
        ),
        ("parse_split_index", True, "test[10]", "test[10:11]"),
        ("parse_split_index", True, "test + train", "test+train"),
        ("parse_split_index", False, "", ValueError),
        ("parse_split_index", False, "test++train", ValueError),
        ("parse_split_index", False, None, ValueError),
        ("parse_split_index", False, 13, ValueError),
        ("parse_split_index", False, [1, 4, 5], ValueError),
        ("parse_split_index", False, "test[10:20:2]", NotImplementedError),
        ("parse_str_slice", True, "[2:5]", (2, 5)),
        ("parse_str_slice", True, "[:5]", (None, 5)),
        ("parse_str_slice", True, "[6:]", (6, None)),
        ("parse_str_slice", True, "[[5: 7] ", (5, 7)),
        ("parse_str_slice", True, ":3", (None, 3)),
        ("parse_str_slice", True, "4:5", (4, 5)),
        ("parse_str_slice", False, "[::3]", ValueError),
        ("parse_str_slice", False, "[5:-1]", ValueError),
        ("parse_str_slice", False, "-10:", ValueError),
        ("parse_str_slice", False, "3:3", ValueError),
        ("parse_str_slice", False, "4:3", ValueError),
    ],
)
def test_dataset_methods(dataset_method, is_valid, input1, input2):
    from armory.data import datasets

    if is_valid:
        assert getattr(datasets, dataset_method)(input1) == input2
    else:
        with pytest.raises(input2):
            getattr(datasets, dataset_method)(input1)


# TODO Below here are just copied from tests/test_docker/test_dataset.py.
#  These still need to be cleaned up


def test_filter_by_index():
    from armory.data import datasets

    ds = datasets.mnist(
        "test", shuffle_files=False, preprocessing_fn=None, framework="tf"
    )
    dataset_size = 10000

    for index in ([], [-4, 5, 6], ["1:3"]):
        with pytest.raises(ValueError):
            datasets.filter_by_index(ds, index, dataset_size)

    ds = datasets.mnist("test", shuffle_files=False, preprocessing_fn=None)
    assert ds.size == dataset_size
    ys = np.hstack([next(ds)[1] for i in range(10)])  # first 10 labels

    for index in (
        [1, 3, 6, 5],
        [0],
        [6, 7, 8, 9, 9, 8, 7, 6],
        list(range(10)),
    ):
        ds = datasets.mnist(
            "test", shuffle_files=False, preprocessing_fn=None, index=index
        )
        index = sorted(set(index))
        assert ds.size == len(index)
        ys_index = np.hstack([y for (x, y) in ds])
        # ys_index = np.hstack([next(ds)[1] for i in range(len(index))])
        assert (ys[index] == ys_index).all()


def test_filter_by_class():
    from armory.data import datasets

    with pytest.raises(ValueError):
        datasets.cifar10("test", shuffle_files=False, class_ids=[])

    ds_filtered = datasets.cifar10("test", shuffle_files=False, class_ids=[3])
    for i, (x, y) in enumerate(ds_filtered):
        assert int(y) == 3
    assert i + 1 == 1000

    ds_filtered = datasets.cifar10("test", shuffle_files=False, class_ids=[2, 7])
    for x, y in ds_filtered:
        assert int(y) in [2, 7]


def test_filter_by_class_and_index():
    from armory.data import datasets

    ds_filtered_by_class = datasets.cifar10(
        "test",
        shuffle_files=False,
        preprocessing_fn=None,
        framework="numpy",
        class_ids=[3],
    )
    num_examples = 10
    xs = np.vstack([next(ds_filtered_by_class)[0] for i in range(10)])

    for index in (
        [1, 3, 6, 5],
        [0],
        [6, 7, 8, 9, 9, 8, 7, 6],
        list(range(num_examples)),
    ):
        ds_filtered_by_class_and_idx = datasets.cifar10(
            "test",
            shuffle_files=False,
            preprocessing_fn=None,
            class_ids=[3],
            index=index,
        )
        index = sorted(set(index))
        assert ds_filtered_by_class_and_idx.size == len(index)
        xs_index = np.vstack([x for (x, y) in ds_filtered_by_class_and_idx])
        assert (xs[index] == xs_index).all()


def test_filter_by_str_slice():
    from armory.data import datasets

    ds = datasets.mnist(
        "test", shuffle_files=False, preprocessing_fn=None, framework="tf"
    )
    dataset_size = 10000

    with pytest.raises(ValueError):
        datasets.filter_by_str_slice(ds, "[10000:]", dataset_size)

    ds = datasets.mnist("test", shuffle_files=False, preprocessing_fn=None)
    assert ds.size == dataset_size
    ys = np.hstack([next(ds)[1] for i in range(10)])  # first 10 labels

    for index, target in (
        ("[:5]", ys[:5]),
        ("[3:8]", ys[3:8]),
        ("[0:5]", ys[0:5]),
    ):
        ds = datasets.mnist(
            "test", shuffle_files=False, preprocessing_fn=None, index=index
        )
        assert ds.size == len(target)
        ys_index = np.hstack([y for (x, y) in ds])
        assert (target == ys_index).all()


def test_parse_split_index_ordering(armory_dataset_dir):
    """
    Ensure that output order is deterministic for multiple splits
    """
    from armory.data import datasets

    index = [5, 37, 38, 56, 111]  # test has max index 9999
    split = "test"
    kwargs = dict(
        epochs=1, batch_size=1, dataset_dir=armory_dataset_dir, shuffle_files=False
    )
    ds = datasets.mnist(split=split, **kwargs)
    fixed_order = []
    for i, (x, y) in enumerate(ds):
        if i in index:
            fixed_order.append(x)
        if i >= max(index):
            break

    sliced_split = f"{split}[{index}]"
    ds = datasets.mnist(split=sliced_split, **kwargs)
    output_x = [x for (x, y) in ds]
    assert len(fixed_order) == len(output_x)
    for x_i, x_j in zip(fixed_order, output_x):
        assert (x_i == x_j).all()


# TODO Talk to David to see why these parameters are used....could we just used the
#  bits from the tests_datasets.py::test_generator_construction
#  David says these can be removed once we do all the asserts
# def test_mnist(armory_dataset_dir):
#     from armory.data import datasets
#     batch_size = 600
#     for split, size in [("train", 60000), ("test", 10000)]:
#         dataset = datasets.mnist(
#             split=split, epochs=1, batch_size=batch_size, dataset_dir=armory_dataset_dir,
#         )
#         assert dataset.size == size
#         assert dataset.batch_size == batch_size
#         assert dataset.batches_per_epoch == (
#             size // batch_size + bool(size % batch_size)
#         )
#
#         x, y = dataset.get_batch()
#         assert x.shape == (batch_size, 28, 28, 1)
#         assert y.shape == (batch_size,)


# def test_cifar(armory_dataset_dir):
#     from armory.data import datasets
#     batch_size = 500
#     for split, size in [("train", 50000), ("test", 10000)]:
#         dataset = datasets.cifar10(
#             split=split, epochs=1, batch_size=batch_size, dataset_dir=armory_dataset_dir,
#         )
#         assert dataset.size == size
#         assert dataset.batch_size == batch_size
#         assert dataset.batches_per_epoch == (
#             size // batch_size + bool(size % batch_size)
#         )
#
#         x, y = dataset.get_batch()
#         assert x.shape == (batch_size, 32, 32, 3)
#         assert y.shape == (batch_size,)


# def test_cifar100(armory_dataset_dir):
#     from armory.data import datasets
#     batch_size = 500
#     for split, size in [("train", 50000), ("test", 10000)]:
#         dataset = datasets.cifar100(
#             split=split, epochs=1, batch_size=batch_size, dataset_dir=armory_dataset_dir,
#         )
#         assert dataset.size == size
#         assert dataset.batch_size == batch_size
#         assert dataset.batches_per_epoch == (
#             size // batch_size + bool(size % batch_size)
#         )
#
#         x, y = dataset.get_batch()
#         assert x.shape == (batch_size, 32, 32, 3)
#         assert y.shape == (batch_size,)


def test_digit(armory_dataset_dir):
    from armory.data import datasets

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
    from armory.data import adversarial_datasets

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
    from armory.data import datasets

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
    from armory.data import datasets

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
    from armory.data import datasets

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
    from armory.data import datasets

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


#
# def test_resisc45(armory_dataset_dir):
#     """
#     Skip test if not locally available
#     """
#     from armory.data import datasets
#     if not os.path.isdir(os.path.join(DATASET_DIR, "resisc45_split", "3.0.0")):
#         pytest.skip("resisc45_split dataset not locally available.")
#
#     for split, size in [("train", 22500), ("validation", 4500), ("test", 4500)]:
#         batch_size = 16
#         epochs = 1
#         dataset = datasets.resisc45(
#             split=split, epochs=epochs, batch_size=batch_size, dataset_dir=armory_dataset_dir,
#         )
#         assert dataset.size == size
#         assert dataset.batch_size == batch_size
#         assert dataset.batches_per_epoch == (
#             size // batch_size + bool(size % batch_size)
#         )
#
#         x, y = dataset.get_batch()
#         assert x.shape == (batch_size, 256, 256, 3)
#         assert y.shape == (batch_size,)
#
#
# def test_resisc10(armory_dataset_dir):
#     from armory.data import datasets
#     for split, size in [("train", 5000), ("validation", 1000), ("test", 1000)]:
#         batch_size = 16
#         epochs = 1
#         dataset = datasets.resisc10(
#             split=split, epochs=epochs, batch_size=batch_size, dataset_dir=armory_dataset_dir,
#         )
#         assert dataset.size == size
#         assert dataset.batch_size == batch_size
#         assert dataset.batches_per_epoch == (
#             size // batch_size + bool(size % batch_size)
#         )
#
#         x, y = dataset.get_batch()
#         assert x.shape == (batch_size, 256, 256, 3)
#         assert y.shape == (batch_size,)


def test_librispeech_adversarial(armory_dataset_dir):
    from armory.data import adversarial_datasets

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
    from armory.data import adversarial_datasets

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
    from armory.data import datasets

    if not os.path.exists(os.path.join(armory_dataset_dir, "coco", "2017", "1.1.0")):
        pytest.skip("coco2017 dataset not downloaded.")

    split_size = 5000
    split = "validation"
    dataset = datasets.coco2017(split=split,)
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
    from armory.data import adversarial_datasets

    split_size = 27
    split = "small"
    dataset = adversarial_datasets.dapricot_dev_adversarial(split=split,)
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
    from armory.data import adversarial_datasets

    split_size = 108
    split = "small"
    dataset = adversarial_datasets.dapricot_test_adversarial(split=split,)
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
    from armory.data import datasets

    dataset = datasets.carla_obj_det_train(split="train")
    assert dataset.size == 4727
    # Testing batch_size > 1
    batch_size = 2
    for modality in ["rgb", "depth", "both"]:
        expected_shape = (
            (batch_size, 600, 800, 6)
            if modality == "both"
            else (batch_size, 600, 800, 3)
        )
        ds_batch_size2 = datasets.carla_obj_det_train(
            split="train", batch_size=batch_size, modality=modality
        )
        x, y = ds_batch_size2.get_batch()
        assert x.shape == expected_shape
        assert len(y) == batch_size
        for label_dict in y:
            assert isinstance(label_dict, dict)
            for obj_key in ["labels", "boxes", "area"]:
                assert obj_key in label_dict


def test_carla_obj_det_dev():
    from armory.data import adversarial_datasets

    ds_rgb = adversarial_datasets.carla_obj_det_dev(split="dev", modality="rgb")
    ds_depth = adversarial_datasets.carla_obj_det_dev(split="dev", modality="depth")
    ds_multimodal = adversarial_datasets.carla_obj_det_dev(split="dev", modality="both")
    for i, ds in enumerate([ds_multimodal, ds_rgb, ds_depth]):
        for x, y in ds:
            if i == 0:
                assert x.shape == (1, 600, 800, 6)
            else:
                assert x.shape == (1, 600, 800, 3)

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
    from armory.data import adversarial_datasets

    ds_rgb = adversarial_datasets.carla_obj_det_test(split="test", modality="rgb")
    ds_depth = adversarial_datasets.carla_obj_det_test(split="test", modality="depth")
    ds_multimodal = adversarial_datasets.carla_obj_det_test(
        split="test", modality="both"
    )
    for i, ds in enumerate([ds_multimodal, ds_rgb, ds_depth]):
        for x, y in ds:
            if i == 0:
                assert x.shape == (1, 600, 800, 6)
            else:
                assert x.shape == (1, 600, 800, 3)

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


def test_carla_video_tracking_dev():
    from armory.data import adversarial_datasets

    dataset = adversarial_datasets.carla_video_tracking_dev(split="dev")
    assert dataset.size == 20
    for x, y in dataset:
        assert x.shape[0] == 1
        assert x.shape[2:] == (600, 800, 3)
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
    from armory.data import adversarial_datasets

    dataset = adversarial_datasets.carla_video_tracking_test(split="test")
    assert dataset.size == 20
    for x, y in dataset:
        assert x.shape[0] == 1
        assert x.shape[2:] == (600, 800, 3)
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
    from armory.data import adversarial_datasets

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


def test_variable_length(armory_dataset_dir):
    """
    Test batches with variable length items using digit dataset
    """
    size = 1350
    batch_size = 4
    from armory.data import datasets

    dataset = datasets.digit(
        split="train", epochs=1, batch_size=batch_size, dataset_dir=armory_dataset_dir,
    )
    assert dataset.batches_per_epoch == (size // batch_size + bool(size % batch_size))

    x, y = dataset.get_batch()
    assert x.dtype == object
    assert x.shape == (batch_size,)
    for x_i in x:
        assert x_i.ndim == 1
        assert 1148 <= len(x_i) <= 18262
    assert y.shape == (batch_size,)


#
# def test_generator(armory_dataset_dir):
#     batch_size = 600
#     from armory.data import datasets
#     for split, size in [("train", 60000)]:
#         dataset = datasets.mnist(
#             split=split, epochs=1, batch_size=batch_size, dataset_dir=armory_dataset_dir,
#         )
#
#         for x, y in dataset:
#             assert x.shape == (batch_size, 28, 28, 1)
#             assert y.shape == (batch_size,)
#             break

#
# def test_numpy_generator(armory_dataset_dir):
#     from armory.data import datasets
#     dataset = datasets.mnist(
#         split="train",
#         epochs=1,
#         batch_size=16,
#         dataset_dir=armory_dataset_dir,
#         framework="numpy",
#     )
#     x, y = dataset.get_batch()
#     assert isinstance(x, np.ndarray)
