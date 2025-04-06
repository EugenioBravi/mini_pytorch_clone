from __future__ import annotations
from typing import NamedTuple, Callable, Optional, Union
import numpy as np


class Dependency(NamedTuple):
    tensor: Tensor
    grad_fn: Callable[[np.ndarray], np.ndarray]


Arrayable = Union[float, list, np.ndarray]


def ensure_array(arrayable: Arrayable) -> np.ndarray:
    if isinstance(arrayable, np.ndarray):
        return arrayable
    else:
        return np.array(arrayable)


class Tensor:
    def __init__(
        self,
        data: Arrayable,
        requires_grad: bool = False,
        depends_on: Optional[list[Dependency]] = None,
    ) -> None:
        self.data = ensure_array(data)
        self.requires_grad = requires_grad
        self.depends_on = depends_on or []
        self.shape = self.data.shape
        self.grad: Optional[Tensor] = None

        if self.requires_grad:
            self.zero_grad()

    def __repr__(self) -> str:
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    def zero_grad(self) -> None:
        self.grad = Tensor(np.zeros_like(self.data, dtype=np.float64))

    def backward(self, grad: Optional[Tensor] = None) -> None:
        assert self.requires_grad, "called backward on non-requires-grad tensor"
        assert self.grad is not None, (
            "Gradient should not be None when requires_grad=True"
        )
        if grad is None:
            if self.shape == ():
                grad = Tensor(1.0)
            else:
                raise RuntimeError("grad must be specified for non-0-tensor")

        self.grad.data += grad.data

        for dependency in self.depends_on:
            backward_grad = dependency.grad_fn(grad.data)
            dependency.tensor.backward(Tensor(backward_grad))

    def sum(self) -> Tensor:
        return tensor_sum(self)


def tensor_sum(t: Tensor) -> Tensor:
    """
    Takes a tensor and returns the 0-tensor that's the sum of all its elements.
    """
    data = t.data.sum()
    requires_grad = t.requires_grad

    if requires_grad:

        def grad_fn(grad: np.ndarray) -> np.ndarray:
            """grad is necessarily a 0-tensor, so each input element contributes that much"""
            return grad * np.ones_like(t.data)

        depends_on = [Dependency(t, grad_fn)]
    else:
        depends_on = []
    return Tensor(data, requires_grad, depends_on)


def add(t1: Tensor, t2: Tensor) -> Tensor:
    data = t1.data + t2.data
    requires_grad = t1.requires_grad or t2.requires_grad
    depends_on: list[Dependency] = []
    if t1.requires_grad:
        grad_fn = make_grad_fn(t1.shape, t2.data, lambda g, x: g)
        depends_on.append(Dependency(t1, grad_fn))

    if t2.requires_grad:
        grad_fn = make_grad_fn(t2.shape, t1.data, lambda g, x: g)
        depends_on.append(Dependency(t2, grad_fn))

    return Tensor(data, requires_grad, depends_on)


def mul(t1: Tensor, t2: Tensor) -> Tensor:
    """
    y = a * b
    have dL/dy
    dL/da = dL/dy * b
    """
    data = t1.data * t2.data
    requires_grad = t1.requires_grad or t2.requires_grad
    depends_on: list[Dependency] = []
    if t1.requires_grad:
        grad_fn = make_grad_fn(t1.shape, t2.data, lambda g, x: g * x)
        depends_on.append(Dependency(t1, grad_fn))

    if t2.requires_grad:
        grad_fn = make_grad_fn(t2.shape, t1.data, lambda g, x: g * x)
        depends_on.append(Dependency(t2, grad_fn))

    return Tensor(data, requires_grad, depends_on)


def make_grad_fn(
    original_shape: tuple, other_data: np.ndarray, chain_rule_fn: Callable
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Args:
        original_shape: Shape of the Tensor.
        other_data: Other Tensor data.
        chain_rule_fn: Function that takes (grad, other_data) and returns modified grad.
        example: chain_rule_fn = lambda g, x: g * x <-- multiplication chain rule
    """

    def grad_fn(grad: np.ndarray) -> np.ndarray:
        grad = chain_rule_fn(grad, other_data)
        # 2. Handle broadcasting if shapes don't match
        if grad.shape != original_shape:
            # Sum over added dimensions (dimensions that didn't exist in original)
            added_dims = grad.ndim - len(original_shape)
            for _ in range(added_dims):
                grad = grad.sum(axis=0)

            # Sum over broadcasted dimensions (where original had size 1)
            for i, dim in enumerate(original_shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)

            # Final reshape if needed (for cases like (3,1,1) -> (3,1))
            if grad.shape != original_shape:
                grad = grad.reshape(original_shape)

        return grad

    return grad_fn


def neg(t: Tensor) -> Tensor:
    data = -t.data
    requires_grad = t.requires_grad
    if requires_grad:
        depends_on = [Dependency(t, lambda x: -x)]
    else:
        depends_on = []
    return Tensor(data, requires_grad, depends_on)
