from art.attacks.evasion import (
    ProjectedGradientDescent,
    CarliniLInfMethod,
    BoundaryAttack,
    AutoAttack,
)


class CascadingAttack(AutoAttack):
    """
    Cascading attack that first tries a PGD attack, then CW, then boundary attack.
    """

    def __init__(self, estimator, **kwargs):
        attack_kwargs = {
            "targeted": kwargs["targeted"],
            "batch_size": kwargs["batch_size"],
        }
        pgd_kwargs = {**attack_kwargs, **kwargs["pgd_kwargs"]}
        self.pgd_attack = ProjectedGradientDescent(estimator, **pgd_kwargs)

        cw_kwargs = {**attack_kwargs, **kwargs["cw_kwargs"]}
        self.cw_attack = CarliniLInfMethod(estimator, **cw_kwargs)

        boundary_kwargs = {**attack_kwargs, **kwargs["boundary_kwargs"]}
        del boundary_kwargs["batch_size"]
        self.boundary_attack = BoundaryAttack(estimator, **boundary_kwargs)

        self.attacks = [self.pgd_attack, self.cw_attack, self.boundary_attack]
        super().__init__(estimator=estimator, attacks=self.attacks, **attack_kwargs)
