import unittest
import numpy as np
from mini_pytorch.tensor import Tensor, mul


class TestTensorSum(unittest.TestCase):
    def test_simple_mul(self):
        t1 = Tensor([1, 2, 3], requires_grad=True)
        t2 = Tensor([4, 5, 6], requires_grad=True)

        t3 = mul(t1, t2)
        t3.backward(Tensor([-1.0, -2.0, -3.0]))

        assert t1.grad.data.tolist() == [-4, -10, -18]
        assert t2.grad.data.tolist() == [-1, -4, -9]

    def test_scalar_mul(self):
        t1 = Tensor([1, 2, 3], requires_grad=True)
        t2 = Tensor(10, requires_grad=True)  # Scalar

        t3 = mul(t1, t2)
        t3.backward(Tensor([1, 1, 1]))
        assert t3.data.tolist() == [10, 20, 30]
        assert t1.grad.data.tolist() == [10, 10, 10]
        assert t2.grad.data.tolist() == 6.0

    def test_broadcast_mul(self):
        t1 = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)  # (2, 3)
        t2 = Tensor([7, 8, 9], requires_grad=True)  # (3,)

        t3 = mul(t1, t2)  # shape (2, 3)
        t3.backward(Tensor([[1, 1, 1], [1, 1, 1]]))

        assert t1.grad.data.tolist() == [[7, 8, 9], [7, 8, 9]]
        assert t2.grad.data.tolist() == [5, 7, 9]

    def test_broadcast_mul2(self):
        t1 = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)  # (2, 3)
        t2 = Tensor([[7, 8, 9]], requires_grad=True)  # (1, 3)

        t3 = mul(t1, t2)
        t3.backward(Tensor([[1, 1, 1], [1, 1, 1]]))

        assert t1.grad.data.tolist() == [[7, 8, 9], [7, 8, 9]]
        assert t2.grad.data.tolist() == [[5, 7, 9]]

    def test_t1_no_requires_grad(self):
        t1 = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=False)  # (2, 3)
        t2 = Tensor([[7, 8, 9]], requires_grad=True)  # (1, 3)

        t3 = mul(t1, t2)
        t3.backward(Tensor([[1, 1, 1], [1, 1, 1]]))

        assert t1.grad is None

    def test_t2_no_requires_grad(self):
        t1 = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)  # (2, 3)
        t2 = Tensor([[7, 8, 9]], requires_grad=False)  # (1, 3)

        t3 = mul(t1, t2)
        t3.backward(Tensor([[1, 1, 1], [1, 1, 1]]))

        assert t2.grad is None

    def test_mul_with_zero(self):
        t1 = Tensor([1, 2, 3], requires_grad=True)
        t2 = Tensor([0, 0, 0], requires_grad=True)

        t3 = mul(t1, t2)
        t3.backward(Tensor([1, 1, 1]))

        assert t1.grad.data.tolist() == [0, 0, 0]
        assert t2.grad.data.tolist() == [1, 2, 3]

    def test_mul_with_negatives(self):
        t1 = Tensor([1, -2, 3], requires_grad=True)
        t2 = Tensor([-4, 5, -6], requires_grad=True)

        t3 = mul(t1, t2)
        t3.backward(Tensor([1, 1, 1]))

        assert t1.grad.data.tolist() == [-4, 5, -6]
        assert t2.grad.data.tolist() == [1, -2, 3]

    def test_single_element_tensor(self):
        t1 = Tensor(5.0, requires_grad=True)
        t2 = Tensor(3.0, requires_grad=True)

        t3 = mul(t1, t2)
        t3.backward(Tensor(1.0))

        assert t1.grad.data.tolist() == 3.0
        assert t2.grad.data.tolist() == 5.0

    def test_broadcasted_mul_gradients(self):
        t1 = Tensor([[1.0], [2.0], [3.0]], requires_grad=True)  # (3, 1)
        t2 = Tensor([10.0, 20.0, 30.0], requires_grad=True)  # (3,)

        t3 = mul(t1, t2)  # → (3, 3)
        t3.backward(Tensor(np.ones_like(t3.data)))

        # t1.grad should sum over columns (axis=1)
        assert t1.grad.data.tolist() == [[60.0], [60.0], [60.0]]

        # t2.grad should sum over rows (axis=0)
        assert t2.grad.data.tolist() == [6.0, 6.0, 6.0]

        # Forward pass check
        assert t3.data.tolist() == [
            [10.0, 20.0, 30.0],
            [20.0, 40.0, 60.0],
            [30.0, 60.0, 90.0],
        ]
