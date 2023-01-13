"""
This module enables loading of different perturbation functions in poisoning
"""

from art.attacks.poisoning import PoisoningAttackBackdoor, perturbations

from armory.utils import triggers


def poison_loader_dlbd(**kwargs):
    poison_type = kwargs["poison_type"]
    if poison_type == "pattern":

        def mod(x):
            return perturbations.add_pattern_bd(x, pixel_value=1)

    elif poison_type == "pixel":

        def mod(x):
            return perturbations.add_single_bd(x, pixel_value=1)

    elif poison_type == "image":
        backdoor_path = kwargs.get("backdoor_path")
        if backdoor_path is None:
            raise ValueError(
                "poison_type 'image' requires 'backdoor_path' kwarg path to image"
            )
        backdoor_path = triggers.get_path(backdoor_path)

        size = kwargs.get("size")
        if size is None:
            raise ValueError("poison_type 'image' requires 'size' kwarg tuple")
        size = tuple(size)
        mode = kwargs.get("mode", "RGB")
        blend = kwargs.get("blend", 0.6)
        base_img_size_x = kwargs.get("base_img_size_x", 48)
        base_img_size_y = kwargs.get("base_img_size_y", 48)
        channels_first = kwargs.get("channels_first", False)
        x_shift = kwargs.get("x_shift", (base_img_size_x - size[0]) // 2)
        y_shift = kwargs.get("y_shift", (base_img_size_y - size[1]) // 2)

        def mod(x):
            return perturbations.insert_image(
                x,
                backdoor_path=backdoor_path,
                size=size,
                mode=mode,
                x_shift=x_shift,
                y_shift=y_shift,
                channels_first=channels_first,
                blend=blend,
                random=False,
            )

    else:
        raise ValueError(f"Unknown poison_type {poison_type}")

    return PoisoningAttackBackdoor(mod)
