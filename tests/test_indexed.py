import math

import pytest
import torch
import torch.nn.functional as F

from sol_attn_amd import IndexedPlan, sol_attention_indexed
from sol_attn_amd.reference import sol_reference

HAS_GPU = torch.version.hip is not None and torch.cuda.is_available()


def fixture():
    rows = torch.stack((torch.arange(87, 152), torch.arange(300, 365)))
    keys = torch.stack((torch.cat((torch.arange(0, 23), torch.arange(70, 198), torch.arange(500, 509))),
                        torch.cat((torch.arange(0, 23), torch.arange(283, 411), torch.arange(500, 509)))))
    return IndexedPlan.create(rows, keys, 509, protected_ranges=((0, 23), (500, 509)), local_tokens=0)


def test_indexed_protection_uses_original_indices():
    p = fixture()
    # Packed key block 2 includes the suffix sink; the last query's local key
    # is in packed block 1, not its same-index block alone.
    assert p.protected_blocks.shape == (2, 2, 3)
    assert p.protected_blocks.all()
    rows = torch.tensor([[400]], dtype=torch.int64)
    keys = torch.arange(128).reshape(1, -1)
    p = IndexedPlan.create(rows, keys, 509, local_tokens=0)
    assert not p.protected_blocks.any()


def test_reference_query_subset_preserves_full_key_moments():
    g = torch.Generator().manual_seed(7)
    q = torch.randn((1, 129, 1, 64), generator=g, dtype=torch.bfloat16)
    k, v = [torch.randn((1, 205, 1, 64), generator=g, dtype=torch.bfloat16) for _ in range(2)]
    full, full_counts = sol_reference(q, k, v, beta=0.0)
    partial, counts = sol_reference(q, k, v, beta=0.0, query_blocks=[0, 2])
    assert torch.equal(partial[:, :64], full[:, :64])
    assert torch.equal(partial[:, 128:], full[:, 128:])
    assert partial[:, 64:128].isnan().all()
    assert (counts[:, 1] == -1).all()
    assert torch.equal(counts[:, [0, 2]], full_counts[:, [0, 2]])


@pytest.mark.parametrize("kind", ["unsorted", "duplicate", "negative", "outside", "dtype", "batch"])
def test_indexed_rejects_invalid_metadata(kind):
    rows = torch.tensor([[1, 2]], dtype=torch.int64)
    keys = torch.arange(8).reshape(1, -1)
    if kind == "unsorted": keys = keys.flip(1)
    if kind == "duplicate": keys[0, 1] = 0
    if kind == "negative": rows[0, 0] = -1
    if kind == "outside": keys[0, -1] = 20
    if kind == "dtype": keys = keys.int()
    if kind == "batch": rows = rows.repeat(2, 1)
    with pytest.raises(ValueError):
        IndexedPlan.create(rows, keys, 20)


@pytest.mark.gpu
@pytest.mark.skipif(not HAS_GPU, reason="AMD ROCm GPU required")
@pytest.mark.parametrize("beta", [-math.inf, math.inf, 1.0])
def test_indexed_mask_pooling_oracle_and_unchanged_inputs(beta):
    g = torch.Generator().manual_seed(719)
    cpu = [torch.randn((1200, 4, 128), generator=g, dtype=torch.bfloat16)[:, ::2] for _ in range(3)]
    rows = torch.stack((torch.arange(231, 361), torch.arange(901, 1031)))
    keys = torch.stack((torch.cat((torch.arange(7), torch.arange(191, 703), torch.arange(1180, 1193))),
                        torch.cat((torch.arange(7), torch.arange(661, 1173), torch.arange(1180, 1193)))))
    p = IndexedPlan.create(rows, keys, 1200, protected_ranges=((0, 7), (1180, 1193)), local_tokens=64)
    gpu = [torch.empty((1200, 4, 128), device="cuda", dtype=torch.bfloat16)[:, ::2].copy_(x) for x in cpu]
    before = [x.clone() for x in gpu]
    actual, stats = sol_attention_indexed(*gpu, p.to("cuda"), beta=beta, return_stats=True)
    ref, counts = sol_reference(cpu[0][rows], cpu[1][keys], cpu[2][keys], beta=beta,
                               local_blocks=-1, protected_blocks=p.protected_blocks)
    torch.testing.assert_close(actual.cpu().float(), ref, atol=0.012, rtol=0.025)
    assert (actual.cpu().float() - ref).norm() / ref.norm() < 0.008
    torch.testing.assert_close(stats.exact_blocks.cpu(), counts, atol=0, rtol=0)
    if beta == -math.inf:
        dense = F.scaled_dot_product_attention(cpu[0][rows].float().transpose(1, 2),
            cpu[1][keys].float().transpose(1, 2), cpu[2][keys].float().transpose(1, 2)).transpose(1, 2)
        torch.testing.assert_close(actual.cpu().float(), dense, atol=0.012, rtol=0.025)
    # Keys not in the declared mask must not affect exact OR approximate output.
    seen = torch.unique(keys)
    excluded = torch.ones(1200, dtype=torch.bool, device="cuda"); excluded[seen.to("cuda")] = False
    gpu[1][excluded] = 80; gpu[2][excluded] = 80
    assert torch.equal(actual, sol_attention_indexed(*gpu, p.to("cuda"), beta=beta))
    assert torch.equal(gpu[0], before[0])
    assert torch.equal(gpu[1][seen], before[1][seen])
    assert torch.equal(gpu[2][seen], before[2][seen])
