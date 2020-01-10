"""
API queries to download and use common datasets.
"""

import os

import numpy as np

import tensorflow_datasets as tfds


def mnist_data() -> (dict, dict):
    """
    returns:
        Tuple of dictionaries containing numpy arrays. Keys are {`image`, `label`}
    """

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


def _git_clone(url, dirpath):
    """
    git clone url from dirpath location
    """
    import subprocess
    try:
        subprocess.check_call(['git', 'clone', url], cwd=dirpath)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"git command not found. Is git installed? {e}")
    except subprocess.CalledProcessError as e:
        raise subprocess.CalledProcessError(f"git clone failed to download: {e}")


def digit(zero_pad=False, rootdir='datasets/external', subdir='free-spoken-digit-dataset/recordings', url='https://github.com/Jakobovski/free-spoken-digit-dataset.git') -> (dict, dict):
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

    rootdir - where the datasets are stored
    subdir - standard folder structur for git repo
    url - standard github url
    """
    from scipy.io import wavfile

    dirpath = os.path.join(rootdir, subdir)
    if not os.path.isdir(dirpath):
        os.makedirs(rootdir, exist_ok=True)
        _git_clone(url, rootdir)

    # TODO: Return generators instead of all data in memory
    sample_rate = 8000
    max_length = 18262
    min_length = 1148
    dtype = np.int16
    train_audio, train_labels = [], []
    test_audio, test_labels = [], []
    for sample in range(50):
        for name in 'jackson', 'nicolas', 'theo', 'yweweler':
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
                    audio = np.hstack([audio, np.zeros(max_length - len(audio), dtype=np.int16)])
                if sample >= 5:
                    train_audio.append(audio)
                    train_labels.append(digit)
                else:
                    test_audio.append(audio)
                    test_labels.append(digit)

    train_ds = {
        'audio': np.array(train_audio),
        'label': np.array(train_labels),
    }
    test_ds = {
        'audio': np.array(test_audio),
        'label': np.array(test_labels),
    }
    return train_ds, test_ds


SUPPORTED_DATASETS = {"mnist": mnist_data, "digit": digit}
