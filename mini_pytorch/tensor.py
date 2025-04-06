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

        def grad_fn1(grad: np.ndarray) -> np.ndarray:
            # sum out added dims
            if grad.shape == t1.data.shape:
                return grad
            added_dims = grad.ndim - t1.data.ndim
            for _ in range(added_dims):
                grad = grad.sum(axis=0)
            # Sum across broadcasted (but non-added dims)
            for i, dim in enumerate(t1.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)
            return grad

        depends_on.append(Dependency(t1, grad_fn1))
    if t2.requires_grad:

        def grad_fn2(grad: np.ndarray) -> np.ndarray:
            # sum out added dims
            if grad.shape == t2.data.shape:
                return grad
            added_dims = grad.ndim - t2.data.ndim
            for _ in range(added_dims):
                grad = grad.sum(axis=0)
            # Sum across broadcasted (but non-added dims)
            for i, dim in enumerate(t2.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)
            return grad

        depends_on.append(Dependency(t2, grad_fn2))

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

        def grad_fn1(grad: np.ndarray) -> np.ndarray:
            # Multiply by t2's data with proper broadcasting
            grad = grad * t2.data

            # Sum over expanded dimensions
            if grad.shape != t1.data.shape:
                # Find axes that were broadcasted in the forward pass
                sum_axes = []
                for i in range(-1, -len(grad.shape) - 1, -1):
                    if i < -len(t1.data.shape) or t1.data.shape[i] == 1:
                        sum_axes.append(i)
                if sum_axes:
                    grad = grad.sum(axis=tuple(sum_axes), keepdims=True)

                # Remove extra dimensions if needed
                if grad.ndim > t1.data.ndim:
                    grad = grad.reshape(t1.data.shape)

            return grad

        depends_on.append(Dependency(t1, grad_fn1))

    if t2.requires_grad:

        def grad_fn2(grad: np.ndarray) -> np.ndarray:
            # Multiply by t1's data with proper broadcasting
            grad = grad * t1.data

            # Sum over expanded dimensions
            if grad.shape != t2.data.shape:
                # Find axes that were broadcasted in the forward pass
                sum_axes = []
                for i in range(-1, -len(grad.shape) - 1, -1):
                    if i < -len(t2.data.shape) or t2.data.shape[i] == 1:
                        sum_axes.append(i)
                if sum_axes:
                    grad = grad.sum(axis=tuple(sum_axes), keepdims=True)

                # Remove extra dimensions if needed
                if grad.ndim > t2.data.ndim:
                    grad = grad.reshape(t2.data.shape)

            return grad

        depends_on.append(Dependency(t2, grad_fn2))

    return Tensor(data, requires_grad, depends_on)
