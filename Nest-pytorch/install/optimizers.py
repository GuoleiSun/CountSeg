from typing import Iterable

from torch import optim
from nest import register


@register
def sgd_optimizer(
    parameters: Iterable, 
    lr: float,
    momentum: float = 0.0, 
    dampening: float = 0.0,
    weight_decay: float = 0.0,
    nesterov: bool = False) -> optim.Optimizer:
    """SGD optimizer.
    """

    return optim.SGD(parameters, lr, momentum, dampening, weight_decay, nesterov)


@register
def adadelta_optimizer(
    parameters: Iterable, 
    lr: float = 1.0,
    rho: float = 0.9,
    eps: float = 1e-6,
    weight_decay: float = 0.0) -> optim.Optimizer:
    """Adadelta optimizer.
    """

    return optim.Adadelta(parameters, lr, rho, eps, weight_decay)
