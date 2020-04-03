import os

import pytest

from armory import paths


@pytest.fixture()
def ensure_armory_dirs(request):
    """
    CI doesn't mount volumes
    """
    saved_model_dir = paths.docker().saved_model_dir
    dataset_dir = paths.docker().dataset_dir
    output_dir = paths.docker().output_dir

    os.makedirs(saved_model_dir, exist_ok=True)
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
