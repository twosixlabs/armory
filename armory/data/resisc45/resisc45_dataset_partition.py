"""
Utility to split main data into three separate folders from original download.

This script is to break the original RAR file into 3 separate folders for
    upload as .tar.gz files into the armory-public s3 bucket. It is not called
    by armory.data.datasets.resisc45()
"""

import os
import shutil

from armory import paths

LABELS = [
    "airplane",
    "airport",
    "baseball_diamond",
    "basketball_court",
    "beach",
    "bridge",
    "chaparral",
    "church",
    "circular_farmland",
    "cloud",
    "commercial_area",
    "dense_residential",
    "desert",
    "forest",
    "freeway",
    "golf_course",
    "ground_track_field",
    "harbor",
    "industrial_area",
    "intersection",
    "island",
    "lake",
    "meadow",
    "medium_residential",
    "mobile_home_park",
    "mountain",
    "overpass",
    "palace",
    "parking_lot",
    "railway",
    "railway_station",
    "rectangular_farmland",
    "river",
    "roundabout",
    "runway",
    "sea_ice",
    "ship",
    "snowberg",
    "sparse_residential",
    "stadium",
    "storage_tank",
    "tennis_court",
    "terrace",
    "thermal_power_station",
    "wetland",
]


def split_data(rootdir=None, full="NWPU-RESISC45"):
    if rootdir is None:
        rootdir = os.path.join(
            paths.HostPaths().dataset_dir, "downloads", "manual", "resisc45"
        )
    train = "train"
    validation = "validation"
    test = "test"
    for folder in train, validation, test:
        for label in LABELS:
            os.makedirs(os.path.join(rootdir, folder, label), exist_ok=True)

    for label in LABELS:
        for i in range(1, 701):
            if i <= 500:
                split = train
            elif i <= 600:
                split = validation
            else:
                split = test
            filename = f"{label}_{i:03}.jpg"
            source = os.path.join(rootdir, full, label, filename)
            target = os.path.join(rootdir, split, label, filename)
            print(f"Copying from {source} to {target}")
            shutil.copyfile(source, target)
