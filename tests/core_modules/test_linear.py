import pytest
import torch
import torch.nn as nn

from core_modules.linear import Embedding, Identity


def test_embedding_construction():
    Embedding(10, 10, padding_idx=0)


def test_identity_output_equals_input_1d():
    x = torch.randn(8)
    out = Identity()(x)
    assert torch.equal(out, x)


def test_identity_output_equals_input_2d():
    x = torch.randn(4, 5)
    out = Identity()(x)
    assert torch.equal(out, x)


def test_identity_output_equals_input_high_dim():
    x = torch.randn(2, 3, 4, 5)
    out = Identity()(x)
    assert torch.equal(out, x)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.int32, torch.bool])
def test_identity_preserves_dtype(dtype):
    x = torch.ones(3, 4, dtype=dtype)
    out = Identity()(x)
    assert out.dtype == dtype
    assert torch.equal(out, x)


def test_identity_returns_same_object():
    x = torch.randn(3, 4)
    out = Identity()(x)
    assert out is x


def test_identity_shares_storage():
    x = torch.randn(3, 4)
    out = Identity()(x)
    assert out.data_ptr() == x.data_ptr()


def test_identity_gradient_flows_through():
    x = torch.randn(3, 4, requires_grad=True)
    out = Identity()(x)
    grad = torch.randn_like(out)
    out.backward(grad)
    assert torch.equal(x.grad, grad)


def test_identity_init_accepts_arbitrary_args():
    Identity(54, unused_arg=True)
    Identity(1, 2, 3, foo="bar", baz=None)


def test_identity_matches_torch_nn_identity():
    x = torch.randn(4, 5, 6)
    ours = Identity()(x)
    theirs = nn.Identity()(x)
    assert torch.equal(ours, theirs)


def test_identity_matches_torch_nn_identity_with_init_args():
    x = torch.randn(2, 3)
    ours = Identity(54, unused_arg=True)(x)
    theirs = nn.Identity(54, unused_arg=True)(x)
    assert torch.equal(ours, theirs)


def test_identity_has_no_parameters():
    assert list(Identity().parameters()) == []
