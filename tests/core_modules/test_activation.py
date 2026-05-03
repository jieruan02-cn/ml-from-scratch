import pytest
import torch
import torch.nn.functional as F

from core_modules.activation import ReLU, relu, relu_


def test_relu_matches_torch():
    x = torch.randn(4, 5)
    assert torch.equal(relu(x), F.relu(x))


def test_relu_negative_zeroed():
    x = torch.tensor([-2.0, -0.1, 0.0, 0.1, 2.0])
    expected = torch.tensor([0.0, 0.0, 0.0, 0.1, 2.0])
    assert torch.equal(relu(x), expected)


def test_relu_out_of_place_does_not_mutate():
    x = torch.tensor([-1.0, 1.0])
    original = x.clone()
    relu(x)
    assert torch.equal(x, original)


def test_relu_inplace_mutates_and_returns_same_tensor():
    x = torch.tensor([-1.0, 2.0])
    out = relu(x, inplace=True)
    assert out is x
    assert torch.equal(x, torch.tensor([0.0, 2.0]))


def test_relu_underscore_mutates():
    x = torch.tensor([-1.0, 2.0])
    out = relu_(x)
    assert out is x
    assert torch.equal(x, torch.tensor([0.0, 2.0]))


def test_relu_gradient():
    # Note: gradient at x=0 is 1 here (clamp semantics), whereas F.relu defines it as 0.
    x = torch.tensor([-1.0, 0.0, 1.0], requires_grad=True)
    relu(x).sum().backward()
    assert torch.equal(x.grad, torch.tensor([0.0, 1.0, 1.0]))


def test_relu_inplace_breaks_autograd_on_leaf():
    x = torch.tensor([-1.0, 1.0], requires_grad=True)
    with pytest.raises(RuntimeError):
        relu(x, inplace=True)


@pytest.mark.parametrize("inplace", [False, True])
def test_relu_module_matches_functional(inplace):
    x = torch.randn(3, 4)
    expected = relu(x.clone(), inplace=False)
    out = ReLU(inplace=inplace)(x.clone())
    assert torch.equal(out, expected)


def test_relu_module_stores_inplace_flag():
    assert ReLU(inplace=True).inplace is True
    assert ReLU().inplace is False


@pytest.mark.parametrize("shape", [(8,), (3, 4), (2, 3, 4, 5)])
def test_relu_shapes(shape):
    x = torch.randn(*shape)
    assert relu(x).shape == x.shape
