"""EPLB simulation loop, aligned with the timing and accounting of upstream `ExpertLoadBalancer`.

Timing (same as upstream):
    step t: measure this step's load with the current layout -> write the counts
    into the ring buffer -> rearrange once the accumulated count reaches `interval`
    (the new layout takes effect from step t+1)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from .policy import (
    initial_physical_to_logical,
    rank_loads,
    rebalance_experts_global,
    resolve_num_extra_experts,
)


@dataclass
class EPLBConfig:
    num_layers: int
    num_experts: int            # logical routed experts
    ep_size: int
    extra_per_gpu: int = 1      # starting redundant experts per GPU
    window: int = 1000          # statistics window, in steps
    interval: int = 10000       # rearrangement interval; 0 disables rearrangement
    first_rearrange_ratio: float = 0.25  # first rearrangement at 1/4 of the interval (upstream behaviour)


@dataclass
class EPLBState:
    config: EPLBConfig
    num_extra: int
    num_physical: int
    capacity: int
    physical_to_logical: np.ndarray
    logical_count: np.ndarray
    window: np.ndarray
    window_index: int = 0
    steps_since_rearrange: int = 0
    rearrangements: int = 0
    rearrange_at: list[int] = field(default_factory=list)


def build_state(cfg: EPLBConfig) -> EPLBState:
    num_extra = resolve_num_extra_experts(cfg.num_experts, cfg.extra_per_gpu * cfg.ep_size, cfg.ep_size)
    num_physical = cfg.num_experts + num_extra
    assert num_physical % cfg.ep_size == 0
    capacity = num_physical // cfg.ep_size

    base = initial_physical_to_logical(cfg.num_experts, num_extra)
    p2l = np.broadcast_to(base, (cfg.num_layers, num_physical)).copy()

    logical_count = np.zeros((cfg.num_layers, cfg.num_experts), dtype=np.int32)
    for logical in base:
        logical_count[:, logical] += 1

    window = np.zeros((cfg.window, cfg.num_layers, cfg.num_experts), dtype=np.int64)
    first_delay = max(1, int(cfg.interval * cfg.first_rearrange_ratio))
    return EPLBState(
        config=cfg,
        num_extra=num_extra,
        num_physical=num_physical,
        capacity=capacity,
        physical_to_logical=p2l,
        logical_count=logical_count,
        window=window,
        steps_since_rearrange=max(0, cfg.interval - first_delay) if cfg.interval else 0,
    )


def step_metrics(counts: np.ndarray, state: EPLBState) -> tuple[float, float]:
    """Return (moe_lb, rank_load_cv). moe_lb = slowest rank / mean rank."""
    loads = rank_loads(counts, state.physical_to_logical, state.logical_count, state.capacity)
    per_rank = loads.sum(axis=0)  # one GPU carries every layer
    mean = per_rank.mean()
    return float(per_rank.max() / mean), float(per_rank.std() / mean)


def run(traffic: Iterable[np.ndarray], cfg: EPLBConfig, mode: str = "eplb") -> dict:
    """Run a full trace.

    mode:
        "static" -- fixed initial layout, never rearranged (the "no EPLB" baseline)
        "eplb"   -- accumulate a window, rearrange every `interval` steps
    """
    state = build_state(cfg)
    lb: list[float] = []
    cv: list[float] = []
    rearrange_times: list[float] = []

    for t, counts in enumerate(traffic):
        m, c = step_metrics(counts, state)
        lb.append(m)
        cv.append(c)

        if mode == "static" or not cfg.interval:
            continue

        state.window[state.window_index] = counts
        state.window_index = (state.window_index + 1) % cfg.window

        state.steps_since_rearrange += 1
        if state.steps_since_rearrange >= cfg.interval:
            state.steps_since_rearrange = 0
            t0 = time.perf_counter()
            p2l, _, lc = rebalance_experts_global(state.window, state.num_physical, state.capacity)
            rearrange_times.append(time.perf_counter() - t0)
            state.physical_to_logical = p2l
            state.logical_count = lc
            state.rearrangements += 1
            state.rearrange_at.append(t)

    return {
        "moe_lb": np.asarray(lb, dtype=np.float32),
        "rank_cv": np.asarray(cv, dtype=np.float32),
        "num_extra": state.num_extra,
        "num_physical": state.num_physical,
        "capacity": state.capacity,
        "rearrangements": state.rearrangements,
        "rearrange_at": state.rearrange_at,
        "rearrange_seconds": rearrange_times,
        "final_mapping": state.physical_to_logical,
    }


def summarize(lb: np.ndarray, warmup: int = 0) -> dict:
    """Steady-state statistics: mean and tail percentiles after the warmup period."""
    x = lb[warmup:]
    return {
        "mean": float(x.mean()),
        "p50": float(np.percentile(x, 50)),
        "p95": float(np.percentile(x, 95)),
        "p99": float(np.percentile(x, 99)),
        "max": float(x.max()),
    }
