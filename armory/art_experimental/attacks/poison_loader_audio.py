from art.attacks.poisoning import PoisoningAttackBackdoor
import librosa
import numpy as np

from armory.utils import triggers

# from art.attacks.poisoning.perturbations.audio_perturbations import (
#    insert_audio_trigger,
#    insert_tone_trigger,
# )

# NOTE: These cached versions are there until performance updates are made to ART
#    As of 1.12.1, insert_audio_trigger and insert_tone_trigger are really slow


class CacheTrigger:
    def __init__(
        self,
        trigger: np.ndarray,
        random: bool = False,
        shift: int = 0,
        scale: float = 0.1,
    ):
        """
        Adds an audio backdoor trigger to a set of audio examples. Works for a single example or a batch of examples.
        :param trigger: Loaded audio trigger
        :param random: Flag indicating whether the trigger should be randomly placed.
        :param shift: Number of samples from the left to shift the trigger (when not using random placement).
        :param scale: Scaling factor for mixing the trigger.
        :return: Backdoored audio.
        """
        self.trigger = trigger
        self.scaled_trigger = self.trigger * scale
        self.random = random
        self.shift = shift
        self.scale = scale

    def insert(self, x: np.ndarray) -> np.ndarray:
        """
        :param x: N x L matrix or length L array, where N is number of examples, L is the length in number of samples.
                  X is in range [-1,1].
        :return: Backdoored audio.
        """
        n_dim = len(x.shape)
        if n_dim == 2:
            return np.array([self.insert(single_audio) for single_audio in x])
        if n_dim != 1:
            raise ValueError("Invalid array shape " + str(x.shape))
        original_dtype = x.dtype
        audio = np.copy(x)
        length = audio.shape[0]
        bd_length = self.trigger.shape[0]
        if bd_length > length:
            raise ValueError("Backdoor audio does not fit inside the original audio.")
        if self.random:
            shift = np.random.randint(length - bd_length)
        else:
            shift = self.shift
        if shift + bd_length > length:
            raise ValueError("Shift + Backdoor length is greater than audio's length.")

        audio[shift : shift + bd_length] += self.scaled_trigger
        return audio.astype(original_dtype)


class CacheAudioTrigger(CacheTrigger):
    def __init__(
        self,
        sampling_rate: int = 16000,
        backdoor_path: str = "whistle.wav",
        duration: float = 1.0,
        **kwargs,
    ):
        """
        :param sampling_rate: Positive integer denoting the sampling rate for x.
        :param backdoor_path: The path to the audio to insert as a trigger.
        :param duration: Duration of the trigger in seconds. Default `None` if full trigger is to be used.
        """
        backdoor_path = triggers.get_path(backdoor_path)
        trigger, bd_sampling_rate = librosa.load(
            backdoor_path, mono=True, sr=None, duration=duration
        )

        if sampling_rate != bd_sampling_rate:
            print(
                f"Backdoor sampling rate {bd_sampling_rate} does not match with the sampling rate provided. "
                "Resampling the backdoor to match the sampling rate."
            )
            trigger, _ = librosa.load(
                backdoor_path, mono=True, sr=sampling_rate, duration=duration
            )
        super().__init__(trigger, **kwargs)


class CacheToneTrigger(CacheTrigger):
    def __init__(
        self,
        sampling_rate: int = 16000,
        frequency: int = 440,
        duration: float = 0.1,
        **kwargs,
    ):
        """
        :param sampling_rate: Positive integer denoting the sampling rate for x.
        :param frequency: Frequency of the tone to be added.
        :param duration: Duration of the tone to be added.
        """
        trigger = librosa.tone(frequency, sr=sampling_rate, duration=duration)
        super().__init__(trigger, **kwargs)


def poison_loader_audio(**kwargs):
    backdoor_kwargs = kwargs.pop("backdoor_kwargs")

    if "backdoor_path" in backdoor_kwargs:

        trigger = CacheAudioTrigger(**backdoor_kwargs)

        def poison_func(x):
            return trigger.insert(x)

        # def poison_func(x):
        #    return insert_audio_trigger(x, **backdoor_kwargs)

    elif "frequency" in backdoor_kwargs:

        trigger = CacheToneTrigger(**backdoor_kwargs)

        def poison_func(x):
            return trigger.insert(x)

        # def poison_func(x):
        #    return insert_tone_trigger(x, **backdoor_kwargs)

    else:
        raise ValueError(
            'backdoor_kwargs should include either "frequency" or "backdoor_path"'
        )
    return PoisoningAttackBackdoor(poison_func, **kwargs)
