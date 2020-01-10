"""
API queries to download and use common datasets.
"""

import os
import zipfile


def mnist_data() -> (dict, dict):
    """
    returns:
        Tuple of dictionaries containing numpy arrays. Keys are {`image`, `label`}
    """
    import tensorflow_datasets as tfds

    # TODO: Return generators instead of all data in memory
    train_ds = tfds.load(
        "mnist", split=tfds.Split.TRAIN, batch_size=-1, data_dir="datasets/"
    )
    test_ds = tfds.load(
        "mnist", split=tfds.Split.TEST, batch_size=-1, data_dir="datasets/"
    )

    return (
        tfds.as_numpy(train_ds),
        tfds.as_numpy(test_ds),
    )


def _curl(url, dirpath, filename):
    """
    git clone url from dirpath location
    """
    import subprocess

    try:
        subprocess.check_call(["curl", "-L", url, "--output", filename], cwd=dirpath)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"curl command not found. Is curl installed? {e}")
    except subprocess.CalledProcessError as e:
        raise subprocess.CalledProcessError(f"curl failed to download: {e}")


def digit(zero_pad=False, rootdir="datasets/external") -> (dict, dict):
    """
    returns:
        Return tuple of dictionaries containing numpy arrays.
        Keys are {`audio`, `label`}
        Sample Rate is 8000 Hz
        Audio samples are of different length, so this returns a numpy object array
            dtype of internal arrays are np.int16
            min length = 1148 samples
            max length = 18262 samples
    
    zero_pad - whether to pad the audio samples to the same length
        if done, this returns `audio` arrays as 2D np.int16 arrays

    rootdir - where the dataset is stored
    """
    import numpy as np
    from scipy.io import wavfile

    url = "https://github.com/Jakobovski/free-spoken-digit-dataset/archive/v1.0.8.zip"
    zip_file = "free-spoken-digit-dataset-1.0.8.zip"
    subdir = "free-spoken-digit-dataset-1.0.8/recordings"

    dirpath = os.path.join(rootdir, subdir)
    if not os.path.isdir(dirpath):
        zip_filepath = os.path.join(rootdir, zip_file)
        # Download file if it does not exist
        if not os.path.isfile(zip_filepath):
            os.makedirs(rootdir, exist_ok=True)
            _curl(url, rootdir, zip_file)

        # Extract and clean up
        with zipfile.ZipFile(zip_filepath, "r") as zip_ref:
            zip_ref.extractall(rootdir)
        os.remove(zip_filepath)

    # TODO: Return generators instead of all data in memory
    # NOTE: issue #21
    sample_rate = 8000
    max_length = 18262
    min_length = 1148
    dtype = np.int16
    train_audio, train_labels = [], []
    test_audio, test_labels = [], []
    for sample in range(50):
        for name in "jackson", "nicolas", "theo":  # , 'yweweler': not yet in release
            for digit in range(10):
                filepath = os.path.join(dirpath, f"{digit}_{name}_{sample}.wav")
                try:
                    s_r, audio = wavfile.read(filepath)
                except FileNotFoundError as e:
                    raise FileNotFoundError(f"digit dataset incomplete. {e}")
                if s_r != sample_rate:
                    raise ValueError(f"{filepath} sample rate {s_r} != {sample_rate}")
                if audio.dtype != dtype:
                    raise ValueError(f"{filepath} dtype {audio.dtype} != {dtype}")
                if zero_pad:
                    audio = np.hstack(
                        [audio, np.zeros(max_length - len(audio), dtype=np.int16)]
                    )
                if sample >= 5:
                    train_audio.append(audio)
                    train_labels.append(digit)
                else:
                    test_audio.append(audio)
                    test_labels.append(digit)

    train_ds = {
        "audio": np.array(train_audio),
        "label": np.array(train_labels),
    }
    test_ds = {
        "audio": np.array(test_audio),
        "label": np.array(test_labels),
    }
    return train_ds, test_ds


SUPPORTED_DATASETS = {"mnist": mnist_data, "digit": digit}
