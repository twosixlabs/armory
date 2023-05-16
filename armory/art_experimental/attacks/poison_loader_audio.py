from art.attacks.poisoning import PoisoningAttackBackdoor

from art.attacks.poisoning.perturbations.audio_perturbations import (
    CacheAudioTrigger,
    CacheToneTrigger,
)


def poison_loader_audio(**kwargs):
    backdoor_kwargs = kwargs.pop("backdoor_kwargs")

    if "backdoor_path" in backdoor_kwargs:
        trigger = CacheAudioTrigger(**backdoor_kwargs)

        def poison_func(x):
            return trigger.insert(x)

    elif "frequency" in backdoor_kwargs:
        trigger = CacheToneTrigger(**backdoor_kwargs)

        def poison_func(x):
            return trigger.insert(x)

    else:
        raise ValueError(
            'backdoor_kwargs should include either "frequency" or "backdoor_path"'
        )
    return PoisoningAttackBackdoor(poison_func, **kwargs)
