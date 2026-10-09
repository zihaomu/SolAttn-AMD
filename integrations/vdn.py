"""Explicit, temporary VDN window-only experiment. Global/linear paths unchanged."""

import time

import torch

from sol_attn_amd import IndexedPlan, sol_attention_indexed


class VDNWindowSOL:
    def __init__(self, policy, beta=1.0, *, base_beta=None, stats=False, base_enabled=True):
        self.policy = policy
        self.beta = beta
        self.base_beta = beta if base_beta is None else base_beta
        self.stats = stats
        self.base_enabled = base_enabled
        self.baseline = type(policy).__call__
        self.cache = {}
        self.calls = 0
        self.setup_seconds = 0.0
        self.density_samples = []

    def prepare(self, layout, bounds, device, anchor):
        if anchor not in ("none", "rows", "columns", "both"):
            raise ValueError("Unknown VDN anchor policy")
        policy = self.policy
        plan = policy.prepare(layout, bounds, device, anchor)
        key = (id(plan), str(device))
        if key not in self.cache:
            tick = time.perf_counter()
            if policy.window_varlen:
                raise ValueError("Research adapter does not support packed varlen")
            if plan.has_windows and policy.batches is None:
                policy.batches = policy._window_batches(plan)
            protected = [(0, layout.video_start), (layout.video_end, layout.seq_len)]
            if anchor in ("columns", "both"):
                for f in (0, layout.num_frames - 1):
                    a = layout.video_start + f * layout.tokens_per_frame
                    protected.append((a, a + layout.tokens_per_frame))
            protected = tuple((int(a), int(b)) for a, b in protected if a < b)
            windows = []
            for rows, keys in policy.batches or []:
                cpu = IndexedPlan.create(rows.cpu(), keys.cpu(), layout.seq_len,
                                         protected_ranges=protected, local_tokens=64)
                windows.append((cpu, cpu.to(device)))
            self.cache[key] = (plan, windows)
            self.setup_seconds += time.perf_counter() - tick
        return self.cache[key]

    def __call__(self, q, k, v, layout, bounds, scale, anchor_frames="none"):
        if not self.base_enabled and layout.tokens_per_frame < 1008:
            return self.baseline(self.policy, q, k, v, layout, bounds, scale, anchor_frames)
        plan, windows = self.prepare(layout, bounds, q.device, anchor_frames)
        out = torch.empty_like(q)
        policy = self.policy
        # Global text/audio and anchor-row queries retain their dense policy.
        if len(plan.dense_q):
            chunk = policy.query_chunk or len(plan.dense_q)
            for start in range(0, len(plan.dense_q), chunk):
                rows = plan.dense_q[start:start + chunk]
                out[rows] = policy.dense(q[rows], k, v, scale)
        beta = self.base_beta if layout.tokens_per_frame < 1008 else self.beta
        exact, total = 0, 0
        for _, indexed in windows:
            result = sol_attention_indexed(q, k, v, indexed, beta=beta, scale=scale,
                                           return_stats=self.stats)
            if self.stats:
                result, stats = result
                exact += stats.exact_blocks.sum().item()
                total += stats.exact_blocks.numel() * stats.key_blocks
            out[indexed.rows] = result
        if self.stats:
            self.density_samples.append(dict(tokens=layout.seq_len, beta=beta,
                                             window_block_density=exact / total if total else 1.0))
        self.calls += 1
        return out


def apply_engine(engine, **kwargs):
    """Explicit per-process override; caller MUST restore the returned token."""
    policy = engine.attention
    kind, old = type(policy), type(policy).__call__
    adapter = VDNWindowSOL(policy, **kwargs)

    def invoke(self, *args, **kw):
        if self is policy:
            return adapter(*args, **kw)
        return old(self, *args, **kw)

    kind.__call__ = invoke
    return kind, old, adapter


def restore_engine(engine, token):
    kind, old, adapter = token
    kind.__call__ = old
    adapter.cache.clear()
