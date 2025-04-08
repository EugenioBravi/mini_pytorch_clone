import unittest
import numpy as np
from mini_pytorch.tensor import Tensor


class TestTensorNeg(unittest.TestCase):
    def test_neg_basic(self):
        """Test basic negation with scalar"""
        t1 = Tensor(5.0, requires_grad=True)
        t2 = -t1

        # Forward pass check
        assert t2.data.tolist() == -5.0

        # Backward pass
        t2.backward()
        assert t1.grad.data.tolist() == -1.0, "Gradient should be -1 (∂(-x)/∂x = -1)"

    def test_neg_vector(self):
        """Test negation with 1D tensor"""
        t1 = Tensor([1.0, -2.0, 3.0], requires_grad=True)
        t2 = -t1

        # Forward pass
        assert t2.data.tolist() == [-1.0, 2.0, -3.0]

        # Backward pass with custom upstream gradient
        upstream_grad = Tensor([0.1, 0.2, 0.3])
        t2.backward(upstream_grad)
        assert t1.grad.data.tolist() == [-0.1, -0.2, -0.3], (
            "Gradient should be negative of upstream grad"
        )

    def test_neg_matrix(self):
        """Test negation with 2D tensor"""
        t1 = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        t2 = -t1

        # Forward pass
        assert t2.data.tolist() == [[-1.0, -2.0], [-3.0, -4.0]]

        # Backward pass with ones
        t2.backward(Tensor(np.ones_like(t2.data)))
        assert t1.grad.data.tolist() == [[-1.0, -1.0], [-1.0, -1.0]]

    def test_neg_chain(self):
        """Test multiple negations in sequence"""
        t1 = Tensor(2.0, requires_grad=True)
        t2 = -t1  # -2.0
        t3 = -t2  # 2.0
        t4 = -t3  # -2.0

        t4.backward()

        assert t4.data.tolist() == -2.0
        assert t1.grad.data.tolist() == (-1) * (-1) * (-1) == -1, (
            "Chain rule: ∂(--x)/∂x = -1 * -1 * -1"
        )

    def test_neg_custom_grad(self):
        """Test negation with non-unity upstream gradient"""
        t1 = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        t2 = -t1

        # Backward pass with custom gradient
        t2.backward(Tensor([2.0, 4.0, 6.0]))

        assert t2.data.tolist() == [-1.0, -2.0, -3.0]
        assert t1.grad.data.tolist() == [-2.0, -4.0, -6.0], (
            "Gradient should be negative of upstream grad"
        )

    def test_neg_mixed_ops(self):
        """Test negation combined with other operations"""
        t1 = Tensor(3.0, requires_grad=True)
        t2 = Tensor(4.0, requires_grad=True)

        # -(3 + 4)
        t3 = -(t1 + t2)
        t3.backward()

        assert t3.data.tolist() == -7.0
        assert t1.grad.data.tolist() == -1.0
        assert t2.grad.data.tolist() == -1.0

    def test_neg_high_dim(self):
        """Test negation with 3D tensor"""
        t1 = Tensor(np.ones((2, 2, 2)), requires_grad=True)
        t2 = -t1

        # Forward pass
        assert t2.data.tolist() == [
            [[-1.0, -1.0], [-1.0, -1.0]],
            [[-1.0, -1.0], [-1.0, -1.0]],
        ]

        # Backward pass
        t2.backward(Tensor(np.ones_like(t2.data)))
        assert t1.grad.data.tolist() == [
            [[-1.0, -1.0], [-1.0, -1.0]],
            [[-1.0, -1.0], [-1.0, -1.0]],
        ]
