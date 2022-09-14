from art.attacks.poisoning import PoisoningAttackBackdoor
from art.attacks.poisoning.perturbations.audio_perturbations import insert_audio_trigger, insert_tone_trigger


def poison_loader_audio(**kwargs):

    backdoor_kwargs = kwargs.pop("backdoor_kwargs")

    if "backdoor_path" in backdoor_kwargs:
        def poison_func(x):
            return insert_audio_trigger(x, **backdoor_kwargs)
    elif "frequency" in backdoor_kwargs:
        def poison_func(x):
            return insert_tone_trigger(x, **backdoor_kwargs)
    else:
        raise ValueError("backdoor_kwargs should include either \"frequency\" or \"backdoor_path\"")


    return PoisoningAttackBackdoor(poison_func, **kwargs)

