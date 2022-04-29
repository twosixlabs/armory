"""
Support for patch attacks using same interface.
"""


from art.attacks.attack import EvasionAttack


class AttackWrapper(EvasionAttack):
    def __init__(self, attack, apply_patch_args, apply_patch_kwargs):
        self._attack = attack
        self.args = apply_patch_args
        self.kwargs = apply_patch_kwargs

    def generate(self, x, y=None, **kwargs):
        self._attack.generate(x, y=y, **kwargs)
        return self._attack.apply_patch(x, *self.args, **self.kwargs)
