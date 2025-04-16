import unittest
import numpy as np
from mini_pytorch.tensor import Tensor


class TestTensorMatmul(unittest.TestCase):
    def test_simple_matmul(self):
        # t1 is (3,2)
        t1 = Tensor([[1, 2], [3, 4], [5, 6]], requires_grad=True)
        # t2 is (2,1)
        t2 = Tensor([[10], [20]], requires_grad=True)
        # t3 is (3,1)
        t3 = t1 @ t2
        assert t3.data.tolist() == [[50], [110], [170]]
        grad = Tensor([[-1.0], [-2.0], [-3.0]])
        t3.backward(grad)

        np.testing.assert_array_equal(t1.grad.data, grad.data @ t2.data.T)
        np.testing.assert_array_equal(t2.grad.data, t1.data.T @ grad.data)

    def test_matmul_no_grad(self):
        # t1 is (3,2)
        t1 = Tensor(
            np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64), requires_grad=False
        )
        # t2 is (2,1)
        t2 = Tensor(np.array([[10], [20]], dtype=np.float64), requires_grad=False)
        # t3 is (3,1)
        t3 = t1 @ t2
        self.assertTrue(np.allclose(t3.data, np.array([[50], [110], [170]])))
        self.assertFalse(t3.requires_grad)

    def test_matmul_one_requires_grad(self):
        # t1 is (3,2)
        t1 = Tensor(
            np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64), requires_grad=True
        )
        # t2 is (2,1)
        t2 = Tensor(np.array([[10], [20]], dtype=np.float64), requires_grad=False)
        # t3 is (3,1)
        t3 = t1 @ t2
        self.assertTrue(np.allclose(t3.data, np.array([[50], [110], [170]])))
        self.assertTrue(t3.requires_grad)

        grad = Tensor(np.array([[-1.0], [-2.0], [-3.0]], dtype=np.float64))
        t3.backward(grad)
        np.testing.assert_array_equal(t1.grad.data, grad.data @ t2.data.T)

    def test_matmul_larger_matrices(self):
        # t1 is (4,3)
        t1 = Tensor(np.random.rand(4, 3), requires_grad=True)
        # t2 is (3,5)
        t2 = Tensor(np.random.rand(3, 5), requires_grad=True)
        # t3 is (4,5)
        t3 = t1 @ t2
        self.assertEqual(t3.shape, (4, 5))
        grad = Tensor(np.random.rand(4, 5))
        t3.backward(grad)

        np.testing.assert_array_equal(t1.grad.data, grad.data @ t2.data.T)
        np.testing.assert_array_equal(t2.grad.data, t1.data.T @ grad.data)

    def test_matmul_no_backward_call(self):
        # t1 is (3,2)
        t1 = Tensor(
            np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64), requires_grad=True
        )
        # t2 is (2,1)
        t2 = Tensor(np.array([[10], [20]], dtype=np.float64), requires_grad=True)

        t1.zero_grad()  # Simulate zero_grad
        t2.zero_grad()

        # t3 is (3,1)
        t3 = t1 @ t2
        self.assertTrue(np.allclose(t3.data, np.array([[50], [110], [170]])))

        # Verify that gradients are not None and are zero matrices after zero_grad but before backward()
        self.assertIsNotNone(t1.grad)
        self.assertIsNotNone(t2.grad)
        self.assertTrue(np.allclose(t1.grad.data, np.zeros_like(t1.data)))
        self.assertTrue(np.allclose(t2.grad.data, np.zeros_like(t2.data)))

    def test_accumulation(self):
        t1 = Tensor(
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64), requires_grad=True
        )
        t2 = Tensor(
            np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64), requires_grad=True
        )

        t3 = t1 @ t2
        t4 = t1 @ t2

        grad = Tensor(np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float64))

        t3.backward(grad)
        t4.backward(grad)

        expected_grad_t1 = grad.data @ t2.data.T + grad.data @ t2.data.T
        expected_grad_t2 = t1.data.T @ grad.data + t1.data.T @ grad.data

        np.testing.assert_allclose(t1.grad.data, expected_grad_t1)
        np.testing.assert_allclose(t2.grad.data, expected_grad_t2)
