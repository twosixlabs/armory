import glob
import os
from itertools import chain
from random import randint

import torch
import dareblopy as db
from PIL import Image
from io import BytesIO
import numpy as np

from armory import paths


def locate_data(dataset_name, dataset_ver, split):
    data_dir = paths.runtime_paths().dataset_dir
    ds_dir = os.path.join(data_dir, dataset_name, dataset_ver)

    return list(glob.glob(f"{ds_dir}/*{split}*.tfrecord*"))


class ImageTFRecordDataSet(torch.utils.data.IterableDataset):
    def __init__(self, dataset_name, dataset_ver, split, epochs):
        self.data_files = locate_data(dataset_name, dataset_ver, split)
        self.features = {
            "label": db.FixedLenFeature([], db.int64),
            "image": db.FixedLenFeature([], db.string),
        }

        self.epochs = epochs

    @staticmethod
    def decode_image(bytes_img):
        return np.array(Image.open(BytesIO(bytes_img[0])))

    def preprocess(self, x):
        label, image = x
        image = self.decode_image(image)

        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)

        return label, image

    def __iter__(self):
        iters = []
        for _ in range(self.epochs):

            # Note this is shuffled by default
            loader = db.ParsedTFRecordsDatasetIterator(
                self.data_files,
                self.features,
                batch_size=1,
                buffer_size=32,
                seed=randint(0, 10000),
            )
            decoded_iter = map(self.preprocess, loader)
            iters.append(decoded_iter)

        return chain(*iters)
