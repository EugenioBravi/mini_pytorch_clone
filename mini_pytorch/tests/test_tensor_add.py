import unittest
from mini_pytorch.tensor import Tensor, add


class TestTensorSum(unittest.TestCase):
    def test_simple_add(self):
        t1 = Tensor([1, 2, 3], requires_grad=True)
        t2 = Tensor([4, 5, 6], requires_grad=True)

        t3 = add(t1, t2)
        t3.backward(Tensor([-1.0, -2.0, -3.0]))

        assert t1.grad.data.tolist() == [-1, -2, -3]
        assert t2.grad.data.tolist() == [-1, -2, -3]

    def test_scalar_addition(self):
        t1 = Tensor([1, 2, 3], requires_grad=True)
        t2 = Tensor(10, requires_grad=True)  # Scalar

        t3 = add(t1, t2)
        t3.backward(Tensor([1, 1, 1]))  # Passing a gradient of [1, 1, 1]
        assert t3.data.tolist() == [11, 12, 13]
        assert t1.grad.data.tolist() == [1, 1, 1]
        assert t2.grad.data.tolist() == 3.0  # Sum of all gradients

    def test_broadcast_add(self):
        # What is broadcasting? A couple of things:
        # If I do t1 + t2 and t1.shape == t2.shape, it's obvious what to do.
        # but I'm also allowed to add 1s to the beginning of either shape.
        #
        # t1.shape == (10, 5), t2.shape == (5,) => t1 + t2, t2 viewed as (1, 5)
        # t2 = [1, 2, 3, 4, 5] => view t2 as [[1, 2, 3, 4, 5]]
        #
        # The second thing I can do, is that if one tensor has a 1 in some dimension,
        # I can expand it
        # t1 as (10, 5) t2 as (1, 5) is [[1, 2, 3, 4, 5]]

        t1 = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)  # (2, 3)
        t2 = Tensor([7, 8, 9], requires_grad=True)  # (3,)

        t3 = add(t1, t2)  # shape (2, 3)
        t3.backward(Tensor([[1, 1, 1], [1, 1, 1]]))

        assert t1.grad.data.tolist() == [[1, 1, 1], [1, 1, 1]]
        assert t2.grad.data.tolist() == [2, 2, 2]

    def test_broadcast_add2(self):
        t1 = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)  # (2, 3)
        t2 = Tensor([[7, 8, 9]], requires_grad=True)  # (1, 3)

        t3 = add(t1, t2)
        t3.backward(Tensor([[1, 1, 1], [1, 1, 1]]))

        assert t1.grad.data.tolist() == [[1, 1, 1], [1, 1, 1]]
        assert t2.grad.data.tolist() == [[2, 2, 2]]

    def test_t1_no_requires_grad(self):
        t1 = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=False)  # (2, 3)
        t2 = Tensor([[7, 8, 9]], requires_grad=True)  # (1, 3)

        t3 = add(t1, t2)
        t3.backward(Tensor([[1, 1, 1], [1, 1, 1]]))

        assert t1.grad is None
        assert t2.grad.data.tolist() == [[2, 2, 2]]

    def test_t2_no_requires_grad(self):
        t1 = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)  # (2, 3)
        t2 = Tensor([[7, 8, 9]], requires_grad=False)  # (1, 3)

        t3 = add(t1, t2)
        t3.backward(Tensor([[1, 1, 1], [1, 1, 1]]))

        assert t1.grad.data.tolist() == [[1, 1, 1], [1, 1, 1]]
        assert t2.grad is None

    def test_addition_with_zero(self):
        t1 = Tensor([1, 2, 3], requires_grad=True)
        t2 = Tensor([0, 0, 0], requires_grad=True)

        t3 = add(t1, t2)
        t3.backward(Tensor([1, 1, 1]))

        assert t1.grad.data.tolist() == [1, 1, 1]  # Unchanged
        assert t2.grad.data.tolist() == [1, 1, 1]  # Should also get gradient

    def test_addition_with_negatives(self):
        t1 = Tensor([1, -2, 3], requires_grad=True)
        t2 = Tensor([-4, 5, -6], requires_grad=True)

        t3 = add(t1, t2)
        t3.backward(Tensor([1, 1, 1]))

        assert t1.grad.data.tolist() == [1, 1, 1]
        assert t2.grad.data.tolist() == [1, 1, 1]

    def test_single_element_tensor(self):
        t1 = Tensor(5.0, requires_grad=True)  # Scalar
        t2 = Tensor(3.0, requires_grad=True)  # Scalar

        t3 = add(t1, t2)
        t3.backward(Tensor(1.0))

        assert t1.grad.data.tolist() == 1.0
        assert t2.grad.data.tolist() == 1.0
