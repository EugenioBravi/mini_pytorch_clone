import numpy as np
from dataclasses import dataclass, field
from typing import NamedTuple, Callable, Optional, Union


class Dependency(NamedTuple):
    tensor: "Tensor"
    grad_fn: Callable[[np.ndarray], np.ndarray]


Arrayable = Union[float, list, np.ndarray]


def ensure_array(arrayable: Arrayable) -> np.ndarray:
    if isinstance(arrayable, np.ndarray):
        return arrayable
    else:
        return np.array(arrayable)


@dataclass
class Tensor:
    data: Arrayable
    requires_grad: bool = False
    depends_on: list[Dependency] = field(default_factory=list)

    def __post_init__(self):
        self.data = ensure_array(self.data)
        self.shape = self.data.shape
        self.grad: Optional["Tensor"] = None
        if self.requires_grad:
            self.zero_grad()

    def __repr__(self) -> str:
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    def zero_grad(self) -> None:
        self.grad = Tensor(np.zeros_like(self.data))

    def backward(self, grad: Optional["Tensor"] = None) -> None:
        assert self.requires_grad, "called backward on non-requires-grad tensor"
        if grad is None:
            if self.shape == ():
                grad = Tensor(1)
            else:
                raise RuntimeError("grad must be specified for non-0-tensor")

        self.grad.data += grad.data

        for dependency in self.depends_on:
            backward_grad = dependency.grad_fn(grad.data)
            dependency.tensor.backward(Tensor(backward_grad))

    def sum(self) -> "Tensor":
        return tensor_sum(self)


def tensor_sum(t: Tensor) -> Tensor:
    """
    Takes a tensor and returns the 0-tensor that's the sum of all its elements.
    """
    data = t.data.sum()
    requieres_grad = t.requires_grad

    if requieres_grad:

        def grad_fn(grad: np.ndarray) -> np.ndarray:
            """grad is necessarily a 0-tensor, so each input element contributes that much"""
            return grad * np.ones_like(t.data)

        depends_on = [Dependency(t, grad_fn)]
    else:
        depends_on = []
    return Tensor(data, requieres_grad, depends_on)
