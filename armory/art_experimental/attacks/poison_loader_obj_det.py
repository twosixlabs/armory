from armory.art_experimental.attacks.poison_loader_dlbd import poison_loader_dlbd
from armory.utils import config_loading

bad_det_attacks = [
    "BadDetRegionalMisclassificationAttack",
    "BadDetGlobalMisclassificationAttack",
    "BadDetObjectGenerationAttack",
    "BadDetObjectDisappearanceAttack",
]


def poison_loader_obj_det(**kwargs):

    backdoor_kwargs = kwargs.pop("backdoor_kwargs")

    backdoor = poison_loader_dlbd(
        **backdoor_kwargs
    )  # loads the PoisoningAttackBackdoor object
    kwargs["backdoor"] = backdoor

    attack_version = kwargs.pop("attack_variant")
    if attack_version not in bad_det_attacks:
        raise ValueError(
            f"'attack_variant' is {attack_version} but should be one of {bad_det_attacks}"
        )

    config = {
        "module": "art.attacks.poisoning",
        "name": attack_version,
        "kwargs": kwargs,
    }

    return config_loading.load(config)
