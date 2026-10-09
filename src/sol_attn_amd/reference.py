"""Small-input independent Torch oracle; not a production fallback.

Builds a virtual attention sequence: selected blocks supply all original tokens;
unselected blocks supply one centroid token with log(valid_length) added to its
logit. Thus one ordinary softmax implements the paper's numerator/denominator.
No online-softmax recurrence or Triton code is reused.
"""

import math
import torch


def _means(x, block, dtype):
    return torch.stack([x[:, i:i + block].float().mean(1)
                        for i in range(0, x.shape[1], block)], 1).to(dtype)


@torch.no_grad()
def sol_reference(q, k, v, *, beta=1.0, scale=None, local_blocks=1,
                  kv_sink_tokens=0, query_sink_tokens=0, block=64,
                  protected_blocks=None, query_blocks=None):
    """FP32 mathematical oracle with explicitly BF16-rounded K/V centroids."""
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    qm = _means(q, block, torch.float32)
    km = _means(k, block, torch.bfloat16).float()
    vm = _means(v, block, torch.bfloat16).float()
    mean = km.mean(1)
    variance = (km.square().mean(1) - mean.square()).clamp_min(0)
    cuts = ((qm * mean[:, None]).sum(-1)
            + beta * (qm.square() * variance[:, None]).sum(-1).sqrt()) * scale
    output = torch.full(q.shape, math.nan, dtype=torch.float32, device=q.device)
    counts = torch.full(qm.shape[:3], -1, dtype=torch.int32, device=q.device)
    # Optional diagnostic subset keeps moments over ALL allowed keys and Q
    # means at their original block origins. Uncomputed outputs remain NaN.
    blocks = range(qm.shape[1]) if query_blocks is None else tuple(query_blocks)
    for batch in range(q.shape[0]):
        for head in range(q.shape[2]):
            for qb in blocks:
                qr = q[batch, qb * block:(qb + 1) * block, head].float()
                proxy = (qr @ km[batch, :, head].T).mean(0) * scale
                selected = proxy > cuts[batch, qb, head]
                if beta == -math.inf:
                    selected[:] = True
                elif beta == math.inf:
                    selected[:] = False
                for kb in range(km.shape[1]):
                    if (abs(qb - kb) <= local_blocks or kb * block < kv_sink_tokens
                            or qb * block < query_sink_tokens
                            or (protected_blocks is not None
                                and protected_blocks[batch, qb, kb])):
                        selected[kb] = True
                logits, values = [], []
                for kb in range(km.shape[1]):
                    start, end = kb * block, min((kb + 1) * block, k.shape[1])
                    if selected[kb]:
                        logits.append(qr @ k[batch, start:end, head].float().T * scale)
                        values.append(v[batch, start:end, head].float())
                    else:
                        logits.append((qr @ km[batch, kb, head])[:, None] * scale
                                      + math.log(end - start))
                        values.append(vm[batch, kb, head][None])
                probabilities = torch.cat(logits, -1).softmax(-1)
                output[batch, qb * block:(qb + 1) * block, head] = (
                    probabilities @ torch.cat(values, 0))
                counts[batch, qb, head] = selected.sum()
    return output, counts
