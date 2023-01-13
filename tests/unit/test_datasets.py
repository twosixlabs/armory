import numpy as np
import pytest
import tensorflow as tf
import torch

from armory.data import datasets

# TODO Right now unit tests only run for `mnist`, `cifar10`, and `resisc10`
#  figure out if this is what we want or if we need more

# Mark all tests in this file as `unit`
pytestmark = pytest.mark.unit


def test_numpy_data_generator():
    NumpyDataGenerator = datasets.NumpyDataGenerator
    m = 100
    n = 10
    x = np.random.random((m, n))
    y = np.arange(m)
    batch_size = 30
    data_generator = NumpyDataGenerator(
        x, y, batch_size=batch_size, drop_remainder=True, shuffle=False
    )
    for i in range(3):
        data_generator.get_batch()
    x_i, y_i = data_generator.get_batch()
    assert (y_i == y[:batch_size]).all()
    assert data_generator.batches_per_epoch == 3

    data_generator = NumpyDataGenerator(
        x, y, batch_size=batch_size, drop_remainder=False, shuffle=False
    )
    for i in range(3):
        data_generator.get_batch()
    x_i, y_i = data_generator.get_batch()
    assert len(x_i) == m % batch_size
    assert data_generator.batches_per_epoch == 4

    data_generator = NumpyDataGenerator(
        x, y, batch_size=batch_size, drop_remainder=True, shuffle=True
    )
    x_i, y_i = data_generator.get_batch()
    assert not (y_i == y[:batch_size]).all()

    for i in range(2):
        data_generator.get_batch()
    x_i_epoch2, y_i_epoch2 = data_generator.get_batch()
    assert not (y_i == y_i_epoch2).all()

    for x, y in [
        (1, [2, 3, 4]),
        ([1, 2], [3, 4, 5]),
    ]:
        with pytest.raises(ValueError):
            NumpyDataGenerator(x, y)


@pytest.mark.parametrize("name", ["cifar10", "mnist", "resisc10"])
@pytest.mark.parametrize("split", ["test", "train", "validation"])
@pytest.mark.parametrize("framework", ["numpy", "tf", "pytorch"])
def test_dataset_generation(
    name, split, framework, armory_dataset_dir, dataset_generator
):
    if split == "validation" and name in ["cifar10", "mnist"]:
        pytest.skip("Dataset: {} Does not have Split: {}".format(name, split))

    # Expected Output Parameters
    parameters = {
        "cifar10": {
            "sizes": {"train": 50000, "test": 10000},
            "shapes": ((32, 32, 3), ()),
        },
        "mnist": {
            "sizes": {"train": 60000, "test": 10000},
            "shapes": ((28, 28, 1), ()),
        },
        "resisc10": {
            "sizes": {"train": 5000, "test": 1000, "validation": 1000},
            "shapes": ((256, 256, 3), ()),
        },
    }

    batch_size = 16
    dataset = dataset_generator(
        name, batch_size, 1, split, framework, dataset_dir=armory_dataset_dir
    )

    x, y = next(iter(dataset))
    typehead = torch if framework == "pytorch" else tf

    assert x.dtype == typehead.uint8
    assert y.dtype == typehead.int64

    assert x.shape == (batch_size,) + parameters[name]["shapes"][0]
    assert y.shape == (batch_size,) + parameters[name]["shapes"][1]

    if framework == "numpy":
        assert dataset.size == parameters[name]["sizes"][split]
        x, y = dataset.get_batch()
        assert x.shape == (batch_size,) + parameters[name]["shapes"][0]
        assert y.shape == (batch_size,) + parameters[name]["shapes"][1]
        assert isinstance(x, np.ndarray)


@pytest.mark.parametrize(
    "name, batch_size, framework1, framework2",
    [
        ("mnist", 10, "tf", "pytorch"),
        ("cifar10", 10, "tf", "pytorch"),
        ("resisc10", 10, "tf", "pytorch"),
    ],
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


@pytest.mark.parametrize("dataset_name", datasets.SUPPORTED_DATASETS.keys())
def test_supported_datasets(dataset_name):
    assert callable(datasets.SUPPORTED_DATASETS[dataset_name])


@pytest.mark.parametrize(
    "name, batch_size, num_epochs, split, framework, shuffle, exp_cnt",
    [
        (
            "mnist",
            10,
            2,
            "test",
            "pytorch",
            True,
            2000,
        ),
    ],
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
