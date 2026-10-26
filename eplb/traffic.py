"""Synthetic MoE routing traffic, generated as a stream (O(1) memory).

Real MoE expert load has two properties this model tries to capture:

1. Static skew: a few experts receive more traffic (variance in gate weights);
2. Dynamic drift: the hot set migrates over time (the prompt mix changes,
   multi-turn conversations shift the context type).

It is not a reconstruction of V4.1's real distribution. The simulator is only
meant to compare the relative behaviour of different strategies.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np


def make_popularity(
    num_layers: int,
    num_experts: int,
    seed: int,
    concentration: float = 1.0,
) -> np.ndarray:
    """Static popularity of each (layer, expert); returns [layers, experts], rows sum to 1."""
    rng = np.random.default_rng(seed)
    logits = rng.normal(0.0, concentration, size=(num_layers, num_experts))
    logits -= logits.max(axis=-1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(axis=-1, keepdims=True)


def traffic_stream(
    num_steps: int,
    num_layers: int,
    num_experts: int,
    experts_per_token: int = 8,
    batch: int = 256,
    seed: int = 0,
    concentration: float = 1.0,
    skew: float = 1.6,
    drift_period: int = 4000,
    drift_strength: float = 0.8,
    hot_fraction: float = 0.05,
) -> Iterator[np.ndarray]:
    """Yield [layers, experts] integer token counts, one step at a time.

    Args:
        experts_per_token: top-k; each token contributes one visit to k experts.
        batch: tokens per step (hundreds for decode, thousands for prefill).
        skew: popularity exponent; >1 sharpens the distribution, because top-k
            selection concentrates more than the raw gate probabilities do.
        drift_period: how often to draw a new hot set; 0 disables drift.
        drift_strength: logit boost applied to the hot set.
        hot_fraction: fraction of experts reheated each round.
    """
    rng = np.random.default_rng(seed)
    base_logits = np.log(make_popularity(num_layers, num_experts, seed, concentration) + 1e-12)
    visits = batch * experts_per_token
    hot_count = max(1, int(num_experts * hot_fraction))
    p = None

    for step in range(num_steps):
        if p is None or (drift_period > 0 and step % drift_period == 0):
            boost = np.zeros((num_layers, num_experts), dtype=np.float32)
            if hot_count > 0:
                hot = rng.choice(num_experts, size=hot_count, replace=False)
                boost[:, hot] = drift_strength
            logits = base_logits + boost
            logits = logits - logits.max(axis=-1, keepdims=True)
            p = np.exp(logits) ** skew
            p = p / p.sum(axis=-1, keepdims=True)

        out = np.empty((num_layers, num_experts), dtype=np.int32)
        for layer in range(num_layers):
            out[layer] = rng.multinomial(visits, p[layer])
        yield out


def make_traffic(**kwargs) -> np.ndarray:
    """Materialise [steps, layers, experts]; only used by small tests."""
    return np.stack(list(traffic_stream(**kwargs)))
