import os
from contextlib import nullcontext
from itertools import product
from tempfile import TemporaryDirectory

import pytest
import torch

import bitsandbytes as bnb
from bitsandbytes import functional as F
from bitsandbytes.autograd import get_inverse_transform_indices, undo_layout
from bitsandbytes.nn.modules import Linear8bitLt


# contributed by Alex Borzunov, see:
# https://github.com/bigscience-workshop/petals/blob/main/tests/test_linear8bitlt.py

@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (7, 5),
    reason="this test requires a turing-generation or newer GPU, see bitsandbytes docs",
)
def test_layout_exact_match():
    x = (torch.randn(14336 * 3, 14336) * 10).to(torch.int8).cuda()
    for tile_size, order in ((8, 32), "col_turing"), ((32, 32), "col_ampere"):
        transform = lambda x: F.transform(x.cuda(), from_order="row", to_order=order)[0].to(x.device)
        tile_indices = get_inverse_transform_indices(transform, tile_size)
        cxb = transform(x)

        torch.cuda.synchronize()
        restored_x = undo_layout(cxb, tile_indices)
        torch.cuda.synchronize()
        assert restored_x.is_contiguous()
        assert torch.all(torch.eq(restored_x, x))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="this test requires a GPU")
def test_linear_no_igemmlt():
    linear = torch.nn.Linear(1024, 3072)
    x = torch.randn(3, 1024, dtype=torch.half)
    linear_custom = Linear8bitLt(
        linear.in_features,
        linear.out_features,
        linear.bias is not None,
        has_fp16_weights=False,
        threshold=6.0,
    )
    linear_custom.state.force_no_igemmlt = True

    linear_custom.weight = bnb.nn.Int8Params(
        linear.weight.data.clone(), requires_grad=False, has_fp16_weights=False
    ).to(linear.weight.dtype)
    linear_custom.bias = linear.bias
    linear_custom = linear_custom.cuda()
    linear = linear.half().cuda()

    x_ref = x.clone().cuda().requires_grad_(True)
    x_ours = x.clone().cuda().requires_grad_(True)
    fx_ref = linear(x_ref).float()
    grad_proj = torch.randn_like(fx_ref)
    (fx_ref * grad_proj).mean().backward()

    fx_ours = linear_custom(x_ours).float()
    (fx_ours * grad_proj).mean().backward()
    assert torch.allclose(fx_ref, fx_ours, atol=0.02)
    assert torch.allclose(x_ref.grad, x_ours.grad, atol=0.01)
    assert not linear_custom.state.has_fp16_weights
    assert linear_custom.state.CB is not None
    assert linear_custom.state.CxB is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="this test requires a GPU")
@pytest.mark.parametrize("has_fp16_weights, serialize_before_forward, deserialize_before_cuda, force_no_igemmlt",
                         list(product([False, True], [False, True], [False, True], [False, True])))
def test_linear_serialization(has_fp16_weights, serialize_before_forward, deserialize_before_cuda, force_no_igemmlt):
    linear = torch.nn.Linear(32, 96)
    x = torch.randn(3, 32, dtype=torch.half)

    linear_custom = Linear8bitLt(
        linear.in_features,
        linear.out_features,
        linear.bias is not None,
        has_fp16_weights=has_fp16_weights,
        threshold=6.0,
    )
    if force_no_igemmlt:
        linear_custom.state.force_no_igemmlt = True

    linear_custom.weight = bnb.nn.Int8Params(
        linear.weight.data.clone(), requires_grad=has_fp16_weights, has_fp16_weights=has_fp16_weights
    )
    linear_custom.bias = linear.bias
    linear_custom = linear_custom.cuda()

    if serialize_before_forward:
        state_dict_8bit = linear_custom.state_dict()

    x_first = x.clone().cuda().requires_grad_(True)
    fx_first = linear_custom(x_first).float()
    grad_proj = torch.randn_like(fx_first)
    (fx_first * grad_proj).mean().backward()

    if not serialize_before_forward:
        state_dict_8bit = linear_custom.state_dict()

    with TemporaryDirectory() as tmpdir:
        state_path_8bit = os.path.join(tmpdir, "state_8bit.pth")
        state_path = os.path.join(tmpdir, "state.pth")

        torch.save(linear.state_dict(), state_path)
        torch.save(state_dict_8bit, state_path_8bit)

        if not has_fp16_weights:
            assert os.path.getsize(state_path_8bit) < 0.5 * os.path.getsize(state_path)

        new_state_dict = torch.load(state_path_8bit)

    new_linear_custom = Linear8bitLt(
        linear.in_features,
        linear.out_features,
        linear.bias is not None,
        has_fp16_weights=has_fp16_weights,
        threshold=6.0,
    )
    if force_no_igemmlt:
        new_linear_custom.state.force_no_igemmlt = True

    if deserialize_before_cuda:
        with nullcontext() if has_fp16_weights else pytest.raises(RuntimeError):
            new_linear_custom.load_state_dict(new_state_dict, strict=True)

    new_linear_custom = new_linear_custom.cuda()

    if not deserialize_before_cuda:
        new_linear_custom.load_state_dict(new_state_dict, strict=True)

    x_second = x.clone().cuda().requires_grad_(True)
    fx_second = new_linear_custom(x_second).float()
    (fx_second * grad_proj).mean().backward()

    # if 8-bit weights were loaded before .cuda, state is incorrect anyway and RuntimeError was raised
    if has_fp16_weights or not deserialize_before_cuda:
        assert torch.allclose(fx_first, fx_second, atol=1e-5)
        assert torch.allclose(x_first.grad, x_second.grad, atol=1e-5)
