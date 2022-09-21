import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from torch import nn
import torchvision
import torchvision.transforms as transforms

from armory.adapt import pytorch_models


TOL = torch.finfo(torch.float32).eps


def cifar10_data(limit=10):
    """
    Return list of cifar10 test batches as PyTorch tensors and the set of classes
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            #            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=1
    )

    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    xy_batches = []
    iterator = iter(loader)
    for i in range(limit):
        xy_batches.append(iterator.next())

    return xy_batches, classes


def imshow(image_batch):
    img = torchvision.utils.make_grid(image_batch)
    plt.imshow(np.transpose(img.detach().numpy(), (1, 2, 0)))
    plt.show()


def cifar10_model(name="baseline"):
    """
    Return a pretrained cifar10 model
    """
    return pytorch_models.load(name=name)


class Attack:
    def __init__(self, model):
        self.model = model

    def generate(self, x, y=None):
        # translate into torch Tensor?

        pass

    def _generate(self, x, *args, **kwargs):
        raise NotImplementedError("Implement in subclasses")


class Clipper:
    """
    Wrapper around torch.clamp
    """

    def __init__(self, min=None, max=None):
        if min is not None and max is not None:
            if max < min:
                raise ValueError("max cannot be greater than min")
        self.min = min
        self.max = max
        # TODO: fix clipping for multiple dimensions

    def __call__(self, x: torch.Tensor):
        return torch.clamp(x, self.min, self.max)


def get_snr_abs(value: float, units: str = "dB"):
    if value != value:
        raise ValueError("SNR value cannot be nan")

    if units.lower() == "db":
        # Map to absolute SNR domain
        return 10 ** (value / 10)
    elif units == "abs":
        if value < 0:
            raise ValueError("Absolute SNR must be nonnegative")
        return value
    else:
        raise ValueError("units must be 'dB' or 'abs'")


def get_shape(shape):
    if isinstance(shape, torch.Tensor):
        shape = shape.shape
    if len(shape) <= 1:
        raise ValueError("shape must have more than one dimension (first dim is batch)")
    if shape[0] != 1:
        raise ValueError("only batch size of 1 currently supported")
    return shape


def random_snr(x_orig: torch.Tensor, epsilon: float, units: str = "dB"):
    """
    Return a (uniform) random vector in SNR epsilon ball
        Use input x_orig to determine original signal strength
    """
    epsilon_snr = get_snr_abs(epsilon, units=units)

    # Map to l2 epsilon
    epsilon_l2 = l2(x_orig) / (epsilon_snr) ** 0.5
    return random_l2(x_orig.shape, epsilon_l2)


def random_l2(shape, epsilon):
    """
    Return a (uniform) random vector in l2 epsilon ball
    """
    shape = get_shape(shape)
    epsilon = float(epsilon)
    if epsilon < 0:
        raise ValueError("epsilon must be >= 0")
    if epsilon != epsilon:
        raise ValueError("epsilon cannot be nan")
    if epsilon == float("inf"):
        raise ValueError("epsilon must be finite")
    if epsilon == 0:
        return torch.zeros(shape)

    rand = torch.randn(shape)
    rand_unit_vector = rand / l2(rand)
    dimension = rand_unit_vector.numel()
    scale = torch.rand(1) ** (1 / dimension)

    return rand_unit_vector * scale


def random_linf(shape, epsilon):
    """
    Return a (uniform) random vector in linf epsilon ball of shape
        First dimension is assumed to be batch
    """
    shape = get_shape(shape)
    epsilon = float(epsilon)
    if epsilon < 0:
        raise ValueError("epsilon must be >= 0")
    if epsilon != epsilon:
        raise ValueError("epsilon cannot be nan")
    if epsilon == float("inf"):
        raise ValueError("epsilon must be finite")
    if epsilon == 0:
        return torch.zeros(shape)

    # shift and scale the uniform distribution from [0, 1] to [-epsilon, epsilon]
    return (2 * torch.rand(shape) - 1) * epsilon


def random_l1(epsilon):
    raise NotImplementedError()


def random_l0(epsilon):
    raise NotImplementedError()


def nan_to_orig(x: Tensor, x_orig: Tensor):
    index = torch.isnan(x)
    if index.any():
        x[index] = x_orig[index]
    return x


def snr_dist(x_orig: Tensor, x: Tensor, units: str = "dB"):
    """
    Determine SNR of x_orig to x - x_orig

    NOTE: This function is not symmetric!
    """
    absolute_snr = (l2(x_orig) / l2(x - x_orig)) ** 2
    if units == "abs":
        return absolute_snr
    elif units == "dB":
        return 10 * torch.log10(absolute_snr)
    else:
        raise ValueError(f"units must be 'abs' or 'dB', not {units}")


def project_snr(
    x: Tensor,
    x_orig: Tensor,
    epsilon: float,
    units: str = "dB",
    safe: bool = False,
    tolerance: float = TOL,
):
    """
    Project `x` into SNR epsilon ball around `x_orig`
        `x` and `x_orig` are considered batches

    For SNR, signal is defined to be `x_orig` and noise is defined as `x - x_orig`

    units - "dB" or "abs"
        dB - epsilon is a value in decibels
        abs - epsilon is an absolute value

    Safe - whether to replace nan with x_orig

    If x_orig is all 0, then SNR is undefined or infinite
    """
    if x.shape[0] != 1:
        raise NotImplementedError("Cannot currently handle batch size > 1")
    if x.shape != x_orig.shape:
        raise ValueError("`x` and `x_orig` must have same shape.")
    epsilon_snr = get_snr_abs(epsilon, units=units)

    if (x_orig.abs().sum() == 0).any():
        if epsilon_snr == 0:
            return x.clone()
        else:
            return x_orig.clone()

    # Map to l2 epsilon
    epsilon_l2 = l2(x_orig) / (epsilon_snr) ** 0.5

    return project_l2(x, x_orig, epsilon_l2, safe=safe, tolerance=tolerance)


def l2(x: Tensor):
    return torch.sqrt((x**2).sum(dim=tuple(range(1, x.ndim)), keepdims=True))


def l2_dist(x: Tensor, x_orig: Tensor):
    return l2(x - x_orig)


def project_l2(x: Tensor, x_orig: Tensor, epsilon: float, safe=False, tolerance=TOL):
    """
    Project `x` into L2 epsilon ball around `x_orig`
        `x` and `x_orig` are considered batches

    Safe - whether to replace nan with x_orig
    """
    if x.shape[0] != 1:
        raise NotImplementedError("Cannot currently handle batch size > 1")
    if x.shape != x_orig.shape:
        raise ValueError("`x` and `x_orig` must have same shape for L2 to be defined.")
    if epsilon < 0:
        raise ValueError("epsilon must be nonnegative")
    if epsilon == 0:
        return x_orig.clone()

    x = x.clone()
    if safe:
        x = nan_to_orig(x, x_orig)

    delta = x - x_orig  # assume x_orig is finite

    # normalize the delta per example in the batch
    denom = l2(delta)
    if (denom <= epsilon).all():
        # already in L2 ball
        return x
    if not torch.isfinite(denom).all():
        # TODO: handle infinite elements here
        raise NotImplementedError("infinite elements in denom of l2 norm")

    delta = (delta / (denom + tolerance)) * epsilon

    x_out = x_orig + delta
    return x_out


def linf(x: Tensor):
    return torch.amax(x, dim=tuple(range(1, x.ndim)), keepdims=True)


def linf_dist(x: Tensor, x_orig: Tensor):
    return linf(x - x_orig)


def project_linf(x: Tensor, x_orig: Tensor, epsilon: float, safe=False):
    """
    Project `x` into Linf epsilon ball around `x_orig`
        `x` and `x_orig` are considered batches

    Safe - whether to replace nan with x_orig
    """
    if x.shape[0] != 1:
        raise NotImplementedError("Cannot currently handle batch size > 1")
    if x.shape != x_orig.shape:
        raise ValueError(
            "`x` and `x_orig` must have same shape for Linf to be defined."
        )
    if epsilon < 0:
        raise ValueError("epsilon must be nonnegative")
    if epsilon == 0:
        return x_orig.clone()

    x_out = torch.clamp(x, min=x_orig - epsilon, max=x_orig + epsilon)
    if safe:
        x_out = nan_to_orig(x_out, x_orig)
    return x_out


def l1(x: Tensor):
    return x.abs().sum(dim=tuple(range(1, x.ndim)), keepdims=True)


def l1_dist(x: Tensor, x_orig: Tensor):
    return l1(x - x_orig)


def project_l1(x: Tensor, x_orig: Tensor, epsilon: float, safe=False):
    raise NotImplementedError("L1 projection is non-unique")


def l0(x: Tensor):
    """
    Note: nans will be treated as different values
    """
    return (x != x).sum(dim=tuple(range(1, x.ndim)), keepdims=True)


def l0_dist(x: Tensor, x_orig: Tensor):
    """
    Note: non-finite values will be treated as different
    """
    return l0(x - x_orig)


def project_l0(
    x: Tensor, x_orig: Tensor, mask: Tensor, epsilon: float = None, safe: bool = False
):
    raise NotImplementedError("L0 projection is non-unique and requires mask")


# class PGD:
#     PROJECTIONS = {
#         "inf": project_linf,
#         "1": project_l1,
#         "2": project_l2,
#     }
#
#     def __init__(self, estimator, norm="inf", clip_min=None, clip_max=None):
#         self.estimator = estimator  # pytorch model?
#         self.norm = norm
#         self.projections = {}
#         pass
#
#     def generate(self, x_orig, *args, **kwargs):
#         self.x_orig = x_orig
#         for i in range(self.num_inits):
#             self.init(x_orig, i=i)
#             for j in range(self.steps):
#                 x = self.x
#                 pred = self.estimator(x)
#                 error = self.loss(pred, self.y_target)
#                 # should probably check loss each time including iter=0
#                 error.backward()
#                 grad = x.grad
#                 grad_norm = normalize_l2(grad)
#                 step = self.eps_step * -grad_norm
#                 # assert norm(step) == eps_step
#                 # OR:
#                 step = self.alpha * -grad
#                 x_new = x + step
#                 # check for weirdness
#                 if self.weird():  # NaN or inf in x_new
#                     self.check_steps()
#                 x_new = self.clip(x_new)
#                 x = x_new
#                 # should we "predict" for best?
#                 self.check_best()
#                 self.early_stop()
#
#     def check_best(self):
#         """
#         Determine whether the current x is best
#         """
#         # if loss < inf_loss and y_predict:
#         #    pass
#
#     def early_stop(self):
#         pass
#
#     def early_stop_inner(self):
#         pass
#
#     def init(self, x, i=0):
#         """
#         Initialize starting point
#
#         i is the current step and is ignored in the base version
#         """
#         if self.random:
#             pass
#         self.x = x
#
#     def get_gradient(self):
#         return self.estimator.get_grad()
#
#     # def project(self, x):
#     #    return self.project_l2(x, self.x_orig, self.epsilon)
#
#     def project(self, x, norm=None):
#         if norm is None:
#             norm = self.norm
#         norm = str(norm)
#         try:
#             projection = self.projections[norm]
#         except KeyError:
#             raise ValueError(f"norm {norm} is not in {list(self.PROJECTIONS) + [None]}")
#         return projection(x, self.x_orig)
#
#     def clip(self, x):
#         return x
#
#     # TODO: insert pytorch optimizers (e.g., Adam)


# class Attack:
#     def __init__(self, model: torch.nn.Module, domain=None, loss_fn=None):
#         if not isinstance(model, torch.nn.Module):
#             raise ValueError(f"{model} is not a torch.nn.Module")
#         self.model = model
#         self.loss_fn = loss_fn
#
#     def __call__(self, x_orig: torch.Tensor, y_true=None) -> torch.Tensor:
#
#         pass


# def PGD_linf(model, epsilon=8/255, project=project_linf, clip=(0, 1), pert=linf, random_init=random_linf_init, Optimizer):
#     pass


# ThreatModel = namedtuple("ThreatModel", ["project", "init", "random", "dist", "norm", "clip"])
# L2ThreatModel = ThreatModel(
#     project=project_l2,
#     init=None,
#     random=None,
#     dist=l2_dist,
#     norm=l2,
#     clip=None,
# )


class PGD_L2:
    def __init__(
        self, model, epsilon=2, eps_step=0.1, min_clip=0, max_clip=1, dist=None
    ):
        pass


class PGD_Linf:
    def __init__(
        self, model, epsilon=8 / 255, eps_step=1 / 255, min_clip=0, max_clip=1
    ):
        self.model = model
        self.proxy = model
        self.epsilon = epsilon
        self.eps_step = eps_step

        # TODO: specify these in init
        self.loss_fn = nn.CrossEntropyLoss()
        self.task_metric = lambda y_pred, y_true: (
            y_pred.detach().argmax(dim=1) == y_true.detach()
        ).sum() / len(y_true)
        self.distance = linf_dist
        self.random = random_linf
        self.project = project_linf
        self.clip = Clipper(min=min_clip, max=max_clip)

    def status(self):
        self.task_acc = self.task_metric(self.y_pred, self.y_true)
        self.pert = self.distance(self.x_orig.detach(), self.x.detach())
        # TODO: verify x in domain

        if (
            self.best_x is None
            or (self.task_acc, self.loss, self.pert) < self.best_value
        ):
            self.best_x = self.x.detach().clone()
            self.best_value = self.task_acc, self.loss, self.pert
            self.new_best = True
            if self.task_acc == 0:
                self.early_stop = True
        else:
            self.new_best = False
        # log gradient size?

        print(
            f"step = {self.i}, accuracy = {self.task_acc}, loss = {self.loss}, dist = {self.pert}, best = {self.new_best}"
        )
        if self.early_stop:
            print("    Stopped early")

    def gradient(self):
        self.y_pred = self.model(self.x)
        self.loss = self.loss_fn(self.y_pred, self.y_target)
        if self.targeted:
            self.loss = -self.loss
        self.loss.backward()

    def update(self):
        self.x.grad = self.x.grad.sign()  # "normalize" gradient for FGSM
        self.optimizer.step()
        with torch.no_grad():
            self.x_temp = self.project(self.x, self.x_orig, self.epsilon)
            self.x_temp = self.clip(self.x_temp)
            self.x.copy_(self.x_temp)  # need to modify the 'x' held in optimizer

    def call_init(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __call__(
        self,
        x_orig: torch.Tensor,
        y_true=None,
        y_target=None,
        x_init="random",
        iters=10,
    ):
        # TODO: verify x_orig in domain
        iters = int(iters)
        if iters < 0:
            raise ValueError("iters must be nonnegative")
        self.call_init(
            early_stop=False,
            best_x=None,
            x_orig=x_orig,
            y_true=y_true,
            y_target=y_target,
            x_init=x_init,
            iters=iters,
        )
        # check that x_orig is in domain
        self.x = x_orig.detach().clone()
        if x_init == "random":
            self.x = self.x + self.random(self.x_orig, epsilon=self.epsilon)
        elif x_init is not None:
            raise ValueError(f"x_init {x_init} must be 'random' or None")

        self.optimizer = torch.optim.SGD(
            [self.x], lr=self.eps_step
        )  # should be part of constructor
        if self.y_target is None:  # targeted
            self.targeted = True
            if self.y_true is None:
                self.y_target = self.model(self.x)
            else:
                self.y_target = self.y_true
        else:
            self.targeted = False

        self.x.requires_grad = True
        for i in range(iters + 1):
            self.i = i
            self.optimizer.zero_grad()
            self.gradient()
            self.status()  # check for early stop, best value, bad values, etc.
            if self.early_stop or self.i >= self.iters:
                break

            self.update()

        return self.best_x


class PGD_Patch(PGD_Linf):
    def __init__(
        self,
        model,
        epsilon=255 / 255,
        eps_step=255 / 255,
        mask_size=(3, 3, 3),
        **kwargs,
    ):
        super().__init__(model, epsilon=epsilon, eps_step=eps_step, **kwargs)
        self.mask_size = mask_size
        self.position = (0, 0, 0)

    def gradient(self):
        self.x = self.x_zero.clone()
        self.x[self.mask] = self.delta
        super().gradient()

    def update(self):
        self.delta.grad = self.delta.grad.sign()  # "normalize" gradient for FGSM
        self.optimizer.step()
        with torch.no_grad():
            # TODO: how to handle projection?
            #  self.x_temp = self.project(self.x, self.x_orig, self.epsilon)
            self.delta_temp = self.clip(self.delta)
            self.delta.copy_(
                self.delta_temp
            )  # need to modify the 'delta' held in optimizer

    def __call__(
        self,
        x_orig: torch.Tensor,
        y_true=None,
        y_target=None,
        x_init=None,
        iters=10,
    ):
        # TODO: verify x_orig in domain
        iters = int(iters)
        if iters < 0:
            raise ValueError("iters must be nonnegative")
        self.call_init(
            early_stop=False,
            best_x=None,
            x_orig=x_orig,
            y_true=y_true,
            y_target=y_target,
            x_init=x_init,
            iters=iters,
        )
        # check that x_orig is in domain
        self.x = x_orig.detach().clone()
        if x_init == "random":
            self.x = self.x + self.random(self.x_orig, epsilon=self.epsilon)
        elif x_init is not None:
            raise ValueError(f"x_init {x_init} must be 'random' or None")

        if self.y_target is None:  # targeted
            self.targeted = True
            if self.y_true is None:
                self.y_target = self.model(self.x)
            else:
                self.y_target = self.y_true
        else:
            self.targeted = False

        # TODO: improve placement of mask
        #    For now, just use upper left
        self.position = (0, 0, 0)
        self.index = (slice(None),) + tuple(
            slice(i, i + j) for (i, j) in zip(self.position, self.mask_size)
        )
        self.mask = torch.zeros(self.x_orig.shape, dtype=bool)
        self.mask[self.index] = True

        self.delta_orig = self.x_orig[self.mask]
        self.x_zero = self.x_orig.clone()
        self.x_zero[self.mask] = 0
        self.delta = self.delta_orig.clone()

        self.optimizer = torch.optim.SGD(
            [self.delta], lr=self.eps_step
        )  # should be part of constructor

        # modify delta (patch) instead of x directly
        self.delta.requires_grad = True
        for i in range(iters + 1):
            self.i = i
            self.optimizer.zero_grad()
            self.gradient()
            self.status()  # check for early stop, best value, bad values, etc.
            if self.early_stop or self.i >= self.iters:
                break

            self.update()

        # TODO: work on random placement of patch

        return self.best_x


# Create graph of input/output variables and connections?

# attack = Attack([
#     Input(x_orig),
#     X_init(x_orig,
#     Output(x_adv),
# ])(epsilon)


class EoTPGD_Linf(PGD_Linf):
    def __init__(self, *args, samples: int = 5, task_samples: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        if samples < 1:
            raise ValueError("samples must be a positive integer")
        self.samples = int(samples)
        if task_samples < 1:
            raise ValueError("task_samples must be a positive integer")
        self.task_samples = int(task_samples)
        # self.task_metric = lambda y_pred, y_true: torch.mean([self.task_metric(
        #    How do you take multiple samples best?

    def gradient(self, *args, **kwargs):
        for i in range(self.samples):
            super().gradient(*args, **kwargs)
        self.x.grad = self.x.grad / self.samples


# Passing Data Objects Approach
#
#
# class AttackState:
#     pass
#
#
# class PGD_Linf_Take2:
#     def __init__(self, model, epsilon=8/255, eps_step=1/255, min_clip=0, max_clip=1):
#         # include more variables in init
#         self.model = model
#         self.proxy = model
#         self.epsilon = epsilon
#         self.eps_step = eps_step
#         self.loss_fn = nn.CrossEntropyLoss()
#         self.task_metric = lambda y_pred, y_true: (y_pred.detach().argmax(dim=1) == y_true.detach()).sum() / len(y_true)
#         self.distance = linf_dist
#         self.random = random_linf
#         self.project = project_linf
#         self.clip = Clipper(min=min_clip, max=max_clip)
#
#     def status(self):
#         self.task_acc = self.task_metric(self.y_pred, self.y_true)
#         self.pert = self.distance(self.x_orig.detach(), self.x.detach())
#         # TODO: verify x in domain
#
#         if self.best is None or (self.task_acc, self.loss, self.pert) < self.best[1:]:
#             self.best = self.x.detach().clone(), self.task_acc, self.loss, self.pert
#             self.new_best = True
#             if self.task_acc == 0:
#                 self.early_stop = True
#         else:
#             self.new_best = False
#         # log gradient size?
#
#         print(f"step = {self.i}, accuracy = {self.task_acc}, loss = {self.loss}, dist = {self.pert}, best = {self.new_best}")
#         if self.early_stop:
#             print("    Stopped early")
#
#     def gradient(self, v):
#         v.y_pred = self.model(v.x)
#         v.loss = self.loss_fn(v.y_pred, v.y_target)
#         if self.targeted:
#             v.loss = -v.loss
#         v.loss.backward()
#
#     def update(self):
#         self.x.grad = self.x.grad.sign()  # "normalize" gradient for FGSM
#         self.optimizer.step()
#         with torch.no_grad():
#             self.x_temp = self.project(self.x, self.x_orig, self.epsilon)
#             self.x_temp = self.clip(self.x_temp)
#             self.x.copy_(self.x_temp)  # need to modify the 'x' held in optimizer
#
#     def call_init(self, **kwargs):
#         for k, v in kwargs.items():
#             setattr(self, k, v)
#
#     def __call__(self, x_orig: torch.Tensor, y_true=None, y_target=None, x_init="random", iters=10):
#         iters = int(iters)
#         if iters < 0:
#             raise ValueError("iters must be nonnegative")
#         self.call_init(early_stop=False, best=None, x_orig=x_orig, y_true=y_true, y_target=y_target, x_init=x_init, iters=iters)
#         # check that x_orig is in domain
#         self.x = x_orig.detach().clone()
#         if x_init == "random":
#             self.x = self.x + self.random(self.x_orig, epsilon=self.epsilon)
#         elif x_init is not None:
#             raise ValueError(f"x_init {x_init} must be 'random' or None")
#
#         self.optimizer = torch.optim.SGD([self.x], lr=self.eps_step)  # should be part of constructor
#         if self.y_target is None:  # targeted
#             self.targeted = True
#             if self.y_true is None:
#                 self.y_target = self.model(self.x)
#             else:
#                 self.y_target = self.y_true
#         else:
#             self.targeted = False
#
#         self.x.requires_grad = True
#         for i in range(iters+1):
#             self.i = i
#             self.optimizer.zero_grad()
#             self.gradient()
#             self.status()  # check for early stop, best value, bad values, etc.
#             if self.early_stop or self.i >= self.iters:
#                 break
#
#             self.update()
#
#         return self.best[0]
