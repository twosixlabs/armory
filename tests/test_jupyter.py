"""
Tests in this file are meant to be run manually inside of a jupyter notebook, e.g.:
    armory launch pytorch --jupyter
"""

import os

import pytest

pytestmark = pytest.mark.jupyter_manual


TEST_DATA_DIR = os.path.join("tests", "test_data")

# NOTE: for image/audio/video tests, you will need to run the IPython call outside of
#     the function in order to view them


def test_jupyter_image(filename="image_sample.png", filedir=TEST_DATA_DIR):
    filepath = os.path.join(filedir, filename)
    import IPython

    IPython.display.Image(filepath)


def test_jupyter_audio(filename="audio_sample.mp3", filedir=TEST_DATA_DIR):
    filepath = os.path.join(filedir, filename)
    import IPython

    IPython.display.Audio(filepath)


def test_jupyter_video(filename="video_sample.mp4", filedir=TEST_DATA_DIR):
    filepath = os.path.join(filedir, filename)
    import IPython

    IPython.display.Video(filepath)


def test_jupyter_tqdm():
    from tqdm.notebook import trange, tqdm
    from time import sleep

    for i in trange(3, desc="1st loop"):
        for j in tqdm(range(100), desc="2nd loop"):
            sleep(0.01)


def test_jupyter_matplotlib():
    from matplotlib import pyplot as plt

    plt.plot([5, 2, 9, 4, 7], [10, 5, 8, 4, 2])
    plt.show()
