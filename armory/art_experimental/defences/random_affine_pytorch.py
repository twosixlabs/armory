from typing import TYPE_CHECKING, List, Optional, Tuple

from art.preprocessing.expectation_over_transformation.pytorch import EoTPyTorch

if TYPE_CHECKING:
    import torch


class EoTRandomAffinePyTorch(EoTPyTorch):
    """
    This module implements EoT of image affine transforms, including rotation, translation and scaling.
    Code is based on https://pytorch.org/vision/stable/_modules/torchvision/transforms/transforms.html#RandomAffine
    """

    def __init__(
        self,
        nb_samples: int,
        clip_values: Tuple[float, float],
        degree: float,
        translate: List[float] = [0.0, 0.0],
        scale: List[float] = [1.0, 1.0],
        apply_fit: bool = False,
        apply_predict: bool = True,
    ) -> None:
        """
        Create an instance of EoTRandomAffinePyTorch.
        :param nb_samples: Number of random samples per input sample.
        :param clip_values: Tuple of float representing minimum and maximum values of input `(min, max)`.
        :param degree: Range of rotation from [-degree, degree]
        :param translate: Normalized range of horizontal and vertical translation
        :param scale: Scaling range (min, max)
        :param apply_fit: True if applied during fitting/training.
        :param apply_predict: True if applied during predicting.
        """
        super().__init__(
            apply_fit=apply_fit,
            apply_predict=apply_predict,
            nb_samples=nb_samples,
            clip_values=clip_values,
        )

        self.degree = degree
        self.degree_range = (
            (-degree, degree) if isinstance(degree, (int, float)) else degree
        )
        self.translate = translate
        self.scale = scale
        self._check_params()

    def _transform(
        self, x: "torch.Tensor", y: Optional["torch.Tensor"], **kwargs
    ) -> Tuple["torch.Tensor", Optional["torch.Tensor"]]:
        """
        Apply random affine transforms to an image.
        :param x: Input samples.
        :param y: Label of the samples `x`.
        :return: Transformed samples and labels.
        """
        import torch
        import torchvision.transforms.functional as F

        img_size = x.shape[:2]

        angle = float(
            torch.empty(1)
            .uniform_(float(self.degree_range[0]), float(self.degree_range[1]))
            .item()
        )

        max_dx = float(self.translate[0] * img_size[1])
        max_dy = float(self.translate[1] * img_size[0])
        tx = int(round(torch.empty(1).uniform_(-max_dx, max_dx).item()))
        ty = int(round(torch.empty(1).uniform_(-max_dy, max_dy).item()))
        translations = (tx, ty)

        scale = float(torch.empty(1).uniform_(self.scale[0], self.scale[1]).item())

        # x needs to have channel first
        x = x.permute(2, 0, 1)
        x = F.affine(
            img=x, angle=angle, translate=translations, scale=scale, shear=(0.0, 0.0)
        )
        x = x.permute(1, 2, 0)

        return torch.clamp(x, min=self.clip_values[0], max=self.clip_values[1]), y

    def _check_params(self) -> None:

        # pylint: disable=R0916
        if not isinstance(self.degree, (int, float)):
            raise ValueError("The argument `degree` has to be a float.")

        if self.degree < 0:
            raise ValueError("The argument `degree` must be positive.")

        if not isinstance(self.translate, list):
            raise ValueError(
                "The argument `translate` has to be a tuple of normalized translation in width and height"
            )

        for t in self.translate:
            if not (0.0 <= t <= 1.0):
                raise ValueError("translation values should be between 0 and 1")

        if not isinstance(self.scale, list):
            raise ValueError(
                "The argument `scale` has to be a tuple of minimum and maximum scaling"
            )

        for s in self.scale:
            if s <= 0:
                raise ValueError("scale values should be positive")
