"""Correctness checks for the EPLB port.

Three kinds of checks:

1. Structural invariants -- every logical expert has at least one replica, every
   GPU holds exactly `capacity` slots, and the two mappings are inverses of each
   other.
2. Objective quality -- the greedy placement is never beaten by random
   placements, and it reaches the optimum on a constructed uniform case.
3. Brute-force comparison -- on a tiny instance, enumerate every replication
   plan and placement to quantify the greedy gap.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eplb.policy import (  # noqa: E402
    rank_loads,
    rebalance_experts_global,
    resolve_num_extra_experts,
)


def check_invariants(num_layers=3, num_experts=16, extra=8, ep=4, steps=32, seed=0):
    rng = np.random.default_rng(seed)
    counts = rng.integers(50, 500, size=(steps, num_layers, num_experts)).astype(np.int64)
    physical = num_experts + extra
    capacity = physical // ep

    p2l, l2p, lc = rebalance_experts_global(counts, physical, capacity)

    assert p2l.shape == (num_layers, physical)
    assert l2p.shape == (num_layers, num_experts, extra + 1)
    assert lc.shape == (num_layers, num_experts)

    # replica count of each logical expert equals logical_count, and sums to physical
    for layer in range(num_layers):
        actual = np.bincount(p2l[layer], minlength=num_experts)
        assert np.array_equal(actual, lc[layer]), "logical_count disagrees with the actual replica count"
        assert actual.sum() == physical
        assert (actual >= 1).all(), "a logical expert has no replica"

    # the two mappings are inverses
    for layer in range(num_layers):
        for logical in range(num_experts):
            slots = l2p[layer, logical]
            slots = slots[slots >= 0]
            assert len(slots) == lc[layer, logical]
            assert (p2l[layer, slots] == logical).all()
    return True


def check_objective_vs_random(num_layers=2, num_experts=12, extra=6, ep=3, steps=24, trials=200, seed=1):
    rng = np.random.default_rng(seed)
    counts = rng.integers(10, 300, size=(steps, num_layers, num_experts)).astype(np.int64)
    physical = num_experts + extra
    capacity = physical // ep

    p2l, _, lc = rebalance_experts_global(counts, physical, capacity)

    def objective(map_, per_layer_counts):
        loads = rank_loads(per_layer_counts, map_, lc, capacity)  # [steps, layers, ranks]
        return loads.max(axis=-1).sum(axis=0).mean()

    greedy = objective(p2l, counts)
    better = 0
    for _ in range(trials):
        shuffled = np.stack([rng.permutation(physical) for _ in range(num_layers)])
        cand = np.take_along_axis(p2l, shuffled, axis=-1)
        if objective(cand, counts) < greedy - 1e-9:
            better += 1
    assert better == 0, f"{better}/{trials} random placements beat the greedy one"
    return True


def check_uniform_optimum(ep=4, num_experts=8, extra=4, steps=1):
    """With equal load per expert, any even placement is optimal; greedy must hit 1.0."""
    counts = np.full((steps, 1, num_experts), 100, dtype=np.int64)
    physical = num_experts + extra
    capacity = physical // ep
    p2l, _, lc = rebalance_experts_global(counts, physical, capacity)
    loads = rank_loads(counts[0], p2l, lc, capacity)
    per_rank = loads.sum(axis=0)
    assert np.allclose(per_rank, per_rank.mean()), f"uneven placement under uniform load: {per_rank}"
    return True


def _compositions(total, parts):
    """Enumerate every way to split `total` into `parts` non-negative integers."""
    if parts == 1:
        yield (total,)
        return
    for i in range(total + 1):
        for rest in _compositions(total - i, parts - 1):
            yield (i,) + rest


def _true_optimum(counts, num_experts, physical, capacity):
    """Brute force over every replication plan and every placement."""
    ep = physical // capacity
    extra = physical - num_experts
    layer_counts = counts[0, 0]
    best = np.inf

    for comp in _compositions(extra, num_experts):
        base = list(range(num_experts))
        for logical, k in enumerate(comp):
            base.extend([logical] * k)
        # load of one replica = tokens of the logical expert / replica count
        value = [layer_counts[base[s]] / (1 + comp[base[s]]) for s in range(physical)]

        seen = set()
        for perm in itertools.permutations(range(physical)):
            groups = tuple(
                sorted(tuple(sorted(perm[i * capacity:(i + 1) * capacity])) for i in range(ep))
            )
            if groups in seen:
                continue
            seen.add(groups)
            peak = max(sum(value[s] for s in g) for g in groups)
            best = min(best, peak)
    return best


def check_greedy_gap(trials=6, seed=7):
    """Tiny brute force: quantify how far the greedy placement is from optimal."""
    rng = np.random.default_rng(seed)
    gaps = []
    for _ in range(trials):
        num_experts, ep, capacity = 4, 3, 2
        physical = ep * capacity
        counts = rng.integers(10, 200, size=(1, 1, num_experts)).astype(np.int64)

        p2l, _, lc = rebalance_experts_global(counts, physical, capacity)
        greedy = rank_loads(counts[0], p2l, lc, capacity).sum(axis=0).max()

        optimal = _true_optimum(counts, num_experts, physical, capacity)
        gaps.append(greedy / optimal - 1.0)

    gaps = np.array(gaps)
    assert (gaps >= -1e-9).all(), "greedy beat the brute-force optimum: the brute force is wrong"
    assert gaps.max() < 0.05, f"greedy is too far from optimal: {gaps}"
    return gaps


def check_resolve_extra():
    assert resolve_num_extra_experts(384, 1 * 64, 64) == 64      # official decode configuration
    assert resolve_num_extra_experts(384, 4 * 8, 8) == 32        # official EP8 configuration
    assert resolve_num_extra_experts(384, 4 * 64, 64) == 256
    assert (384 + resolve_num_extra_experts(384, 4 * 64, 64)) % 64 == 0
    return True


def main():
    checks = [
        ("structural invariants", check_invariants),
        ("greedy >= random", check_objective_vs_random),
        ("uniform load is perfectly balanced", check_uniform_optimum),
        ("redundancy rounding", check_resolve_extra),
    ]
    for name, fn in checks:
        fn()
        print(f"  PASS  {name}")

    gaps = check_greedy_gap()
    print(f"  PASS  brute force: greedy is {gaps.mean()*100:.2f}% off optimal on average, "
          f"{gaps.max()*100:.2f}% worst case")


if __name__ == "__main__":
    main()
