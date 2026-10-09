import math

import pytest
import torch
import torch.nn.functional as F

from sol_attn_amd import dense_attention, sol_attention
from sol_attn_amd.reference import sol_reference

HAS_GPU = torch.version.hip is not None and torch.cuda.is_available()


def tensors(nq=193, nk=317, d=64, b=1, h=2, device="cpu", strided=False):
    gen = torch.Generator(device=device).manual_seed(42)
    xs = [torch.randn((b, n, h * (2 if strided else 1), d), generator=gen,
                      device=device, dtype=torch.bfloat16) for n in (nq, nk, nk)]
    return tuple(x[:, :, ::2] if strided else x for x in xs)


def assert_close(actual, expected):
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.float(), expected.float(), atol=0.012, rtol=0.025)
    relative_l2 = (actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-8)
    assert relative_l2 < 0.008


@pytest.mark.parametrize("d", [64, 128])
def test_reference_dense_limit(d):
    q, k, v = tensors(d=d)
    out, counts = sol_reference(q, k, v, beta=-math.inf)
    expected = F.scaled_dot_product_attention(q.float().transpose(1, 2),
        k.float().transpose(1, 2), v.float().transpose(1, 2)).transpose(1, 2)
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=2e-5)
    assert (counts == 5).all()


def test_reference_constant_keys_correction():
    q, k, v = tensors()
    k[:] = k[:, :1].clone()
    sparse, _ = sol_reference(q, k, v, beta=math.inf)
    # BF16 centroid rounding is explicit; this is not a bitwise dense claim.
    dense, _ = sol_reference(q, k, v, beta=-math.inf)
    torch.testing.assert_close(sparse, dense, atol=0.001, rtol=0.04)


def test_reference_prefix_protection_rounds_outward():
    q, k, v = tensors()
    _, counts = sol_reference(q, k, v, beta=math.inf,
                             kv_sink_tokens=65, query_sink_tokens=1, local_blocks=0)
    assert (counts[:, 0] == 5).all()
    assert (counts[:, 1] == 2).all()
    assert (counts[:, 2:] == 3).all()


@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("beta", [-math.inf, math.inf, 1.0])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.gpu
@pytest.mark.skipif(not HAS_GPU, reason="AMD ROCm GPU required")
def test_sol_ragged_cross_attention(d, beta, strided):
    q, k, v = tensors(d=d, b=2, device="cuda", strided=strided)
    actual, stats = sol_attention(q, k, v, beta=beta, return_stats=True)
    reference, counts = sol_reference(q.cpu(), k.cpu(), v.cpu(), beta=beta)
    assert_close(actual.cpu(), reference)
    # Routing on finite beta uses WMMA, so near-threshold decisions may differ
    # in general. This fixed-seed fixture is a regression contract.
    torch.testing.assert_close(stats.exact_blocks.cpu(), counts, atol=0, rtol=0)
    repeated = sol_attention(q, k, v, beta=beta)
    assert torch.equal(actual, repeated)


@pytest.mark.gpu
@pytest.mark.skipif(not HAS_GPU, reason="AMD ROCm GPU required")
@pytest.mark.parametrize("nq,nk", [(1, 1), (63, 65), (64, 64), (65, 63), (1025, 1089)])
def test_dense_tail(nq, nk):
    q, k, v = tensors(nq=nq, nk=nk, d=128, device="cuda")
    actual = dense_attention(q, k, v)
    expected = F.scaled_dot_product_attention(q.float().transpose(1, 2),
        k.float().transpose(1, 2), v.float().transpose(1, 2)).transpose(1, 2)
    assert_close(actual, expected)


@pytest.mark.gpu
@pytest.mark.skipif(not HAS_GPU, reason="AMD ROCm GPU required")
def test_sol_zero_inputs_and_exact_sinks():
    q, k, v = tensors(device="cuda")
    q.zero_(); k.zero_()
    actual, stats = sol_attention(q, k, v, beta=math.inf, local_blocks=0,
        kv_sink_tokens=65, query_sink_tokens=1, return_stats=True)
    ref, counts = sol_reference(q.cpu(), k.cpu(), v.cpu(), beta=math.inf,
        local_blocks=0, kv_sink_tokens=65, query_sink_tokens=1)
    assert_close(actual.cpu(), ref)
    torch.testing.assert_close(stats.exact_blocks.cpu(), counts, atol=0, rtol=0)


@pytest.mark.gpu
@pytest.mark.skipif(not HAS_GPU, reason="AMD ROCm GPU required")
def test_sol_single_token_no_empty_mass_nan():
    q, k, v = tensors(nq=1, nk=1, device="cuda")
    for beta in (-math.inf, math.inf, 1.0):
        assert torch.equal(sol_attention(q, k, v, beta=beta), v)


@pytest.mark.gpu
@pytest.mark.skipif(not HAS_GPU, reason="AMD ROCm GPU required")
@pytest.mark.parametrize("beta", [-math.inf, math.inf, 1.0])
def test_sol_multiple_groups_and_tail(beta):
    q, k, v = tensors(nq=1025, nk=1089, h=1, device="cuda")
    actual, stats = sol_attention(q, k, v, beta=beta, return_stats=True)
    ref, counts = sol_reference(q.cpu(), k.cpu(), v.cpu(), beta=beta)
    assert_close(actual.cpu(), ref)
    torch.testing.assert_close(stats.exact_blocks.cpu(), counts, atol=0, rtol=0)


@pytest.mark.gpu
@pytest.mark.skipif(not HAS_GPU, reason="AMD ROCm GPU required")
def test_sol_rows_with_no_exact_blocks():
    q, k, v = tensors(nq=321, nk=65, h=1, device="cuda")
    actual, stats = sol_attention(q, k, v, beta=math.inf, local_blocks=0, return_stats=True)
    ref, counts = sol_reference(q.cpu(), k.cpu(), v.cpu(), beta=math.inf, local_blocks=0)
    assert (stats.exact_blocks[:, 2:] == 0).all()
    assert_close(actual.cpu(), ref)
    torch.testing.assert_close(stats.exact_blocks.cpu(), counts, atol=0, rtol=0)


@pytest.mark.gpu
@pytest.mark.skipif(not HAS_GPU, reason="AMD ROCm GPU required")
def test_sol_preserves_inputs_and_rejects_masks():
    q, k, v = tensors(device="cuda")
    copies = tuple(x.clone() for x in (q, k, v))
    sol_attention(q, k, v)
    for x, copy in zip((q, k, v), copies):
        assert torch.equal(x, copy)
    with pytest.raises(TypeError):
        sol_attention(q, k, v, attn_mask=torch.ones((193, 317), device="cuda"))


@pytest.mark.gpu
@pytest.mark.skipif(not HAS_GPU, reason="AMD ROCm GPU required")
def test_sol_rejects_invalid_controls():
    q, k, v = tensors(device="cuda")
    for controls in ({"beta": math.nan}, {"local_blocks": -1},
                     {"kv_sink_tokens": 318}, {"query_sink_tokens": 194},
                     {"scale": 0}, {"scale": math.inf}):
        with pytest.raises(ValueError):
            sol_attention(q, k, v, **controls)


def test_api_rejects_unsupported_input():
    q, k, v = tensors()
    with pytest.raises(ValueError, match="AMD ROCm GPU"):
        sol_attention(q, k, v)
    with pytest.raises(ValueError, match="Only BF16"):
        sol_attention(q.float(), k.float(), v.float())
    with pytest.raises(ValueError, match="Inference only"):
        sol_attention(q.requires_grad_(), k, v)
