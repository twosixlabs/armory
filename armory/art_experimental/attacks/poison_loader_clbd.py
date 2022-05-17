"""
This module enables loading of CLBD attack from a json config
"""


from art.attacks.poisoning import PoisoningAttackCleanLabelBackdoor
from art.utils import to_categorical

from armory.art_experimental.attacks.poison_loader_dlbd import poison_loader_dlbd


def poison_loader_clbd(**kwargs):
    backdoor_kwargs = kwargs.pop("backdoor_kwargs")
    backdoor = poison_loader_dlbd(**backdoor_kwargs)

    # Targets is a one-hot numpy array -- need to map from sparse representation
    target = kwargs.pop("target")
    n_classes = kwargs.pop("n_classes")
    targets = to_categorical([target], n_classes)[0]

    return (
        PoisoningAttackCleanLabelBackdoor(backdoor=backdoor, target=targets, **kwargs),
        backdoor,
    )
