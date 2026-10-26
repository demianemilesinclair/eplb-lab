"""NumPy port of DeepInfer's EPLB policy.

Ported from `deepinfer/model_executor/eplb/policy.py` in DeepSeek-AI/DeepInfer
(Apache-2.0). This file keeps the original objective functions and greedy
ordering; only the tensor backend changed (torch -> NumPy) so the algorithm runs
on CPU-only machines.

Terminology (same as upstream):
- logical expert: an expert as defined by the model, abbreviated "log";
- physical expert: an expert replica actually placed on a GPU, count =
  logical + redundant;
- capacity: physical experts per GPU ("num_local_physical_experts" upstream).
"""

from __future__ import annotations

import numpy as np


def resolve_num_extra_experts(num_routed_experts: int, min_num_extra_experts: int, ep_size: int) -> int:
    """Round the minimum number of redundant experts up so `physical % ep_size == 0`.

    Mirrors the upstream function of the same name. Upstream writes the padding
    as `-(a + b) % ep`, which is exactly Python's modulo.
    """
    padding = -(num_routed_experts + min_num_extra_experts) % ep_size
    return min_num_extra_experts + padding


def initial_physical_to_logical(num_routed_experts: int, num_extra_experts: int) -> np.ndarray:
    """Initial layout: [original routed experts, duplicated routed experts]."""
    head = np.arange(num_routed_experts, dtype=np.int32)
    tail = (np.arange(num_extra_experts, dtype=np.int32) % num_routed_experts)
    return np.concatenate([head, tail])


def _batch_greedy_indices(
    tokens_per_expert: np.ndarray,        # [steps, layers, logical]
    physical_to_logical_map: np.ndarray,  # [layers, physical]
    logical_count: np.ndarray,            # [layers, logical]
    capacity: int,                        # physical / ep_size
) -> np.ndarray:
    """Place physical experts onto ranks greedily; returns a slot permutation.

    Objective: with the replication plan fixed, minimise
    `sum over steps of (max over ranks of load)`.
    """
    num_steps, num_layers, _ = tokens_per_expert.shape
    num_physical = physical_to_logical_map.shape[1]
    num_ranks = num_physical // capacity

    # Load of one replica = tokens of its logical expert / number of replicas
    vectors = np.transpose(tokens_per_expert, (1, 2, 0)) / logical_count[:, :, None]  # [layers, logical, steps]

    # Heaviest experts first (LPT-style): better greedy results
    weights = tokens_per_expert.sum(axis=0) / logical_count  # [layers, logical]
    physical_loads = np.take_along_axis(weights, physical_to_logical_map, axis=-1)
    physical_order = np.argsort(-physical_loads, axis=-1, kind="stable")  # [layers, physical]
    expert_order = np.take_along_axis(physical_to_logical_map, physical_order, axis=-1)

    loads = np.zeros((num_layers, num_ranks, num_steps), dtype=np.float32)
    peaks = np.zeros((num_layers, num_steps), dtype=np.float32)
    occupancy = np.zeros((num_layers, num_ranks), dtype=np.int64)
    indices = np.empty_like(physical_order)

    layer_ids = np.arange(num_layers)
    for job_index in range(num_physical):
        logical_of_job = expert_order[:, job_index]
        vector = vectors[layer_ids, logical_of_job]                    # [layers, steps]

        candidates = np.maximum(loads + vector[:, None, :], peaks[:, None, :])
        scores = candidates.sum(axis=-1)                               # [layers, ranks]
        scores = np.where(occupancy == capacity, np.inf, scores)
        rank = scores.argmin(axis=-1)

        slot = rank * capacity + occupancy[layer_ids, rank]
        indices[layer_ids, slot] = physical_order[:, job_index]

        updated = loads[layer_ids, rank] + vector
        loads[layer_ids, rank] = updated
        peaks = np.maximum(peaks, updated)
        occupancy[layer_ids, rank] += 1

    return indices


def rebalance_experts_global(
    tokens_per_expert: np.ndarray,        # [steps, layers, logical]
    num_physical_experts: int,
    num_local_physical_experts: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Global balancing: replicate hot experts, then spread replicas across ranks.

    Returns (physical_to_logical_map, logical_to_physical_map, logical_count).
    """
    _, num_moe_layers, num_logical_experts = tokens_per_expert.shape
    num_redundant = num_physical_experts - num_logical_experts
    assert num_redundant >= 0

    physical_to_logical = np.empty((num_moe_layers, num_physical_experts), dtype=np.int32)
    logical_to_physical = np.full(
        (num_moe_layers, num_logical_experts, num_redundant + 1), -1, dtype=np.int32
    )
    logical_count = np.ones((num_moe_layers, num_logical_experts), dtype=np.int32)

    # Initial layout: logical experts in the first slots, redundant replicas after
    arange_logical = np.arange(num_logical_experts, dtype=np.int32)
    physical_to_logical[:, :num_logical_experts] = arange_logical
    logical_to_physical[:, :, 0] = arange_logical

    # Greedy replication: each round picks the expert whose extra replica lowers
    # the global max per-copy load the most.
    tokens_diff = tokens_per_expert + arange_logical * 1e-4  # tie-break, see upstream note
    for i in range(num_redundant):
        score = tokens_diff / logical_count[None, :, :]                 # [steps, layers, logical]
        score_added = tokens_per_expert / (logical_count + 1)[None, :, :]

        argmax_idx = score.argmax(axis=-1, keepdims=True)               # [steps, layers, 1]
        values = np.take_along_axis(score, argmax_idx, axis=-1)
        values = np.broadcast_to(values, score.shape).copy()
        replaced = np.take_along_axis(score_added, argmax_idx, axis=-1)
        np.put_along_axis(values, argmax_idx, replaced, axis=-1)

        redundancy_indices = values.sum(axis=0).argmin(axis=-1)         # [layers]
        physical_index = num_logical_experts + i
        physical_to_logical[:, physical_index] = redundancy_indices

        counts = logical_count[np.arange(num_moe_layers), redundancy_indices]
        logical_to_physical[np.arange(num_moe_layers), redundancy_indices, counts] = physical_index
        logical_count[np.arange(num_moe_layers), redundancy_indices] += 1

    if num_local_physical_experts > 1:
        assert num_physical_experts % num_local_physical_experts == 0
        indices = _batch_greedy_indices(
            tokens_per_expert, physical_to_logical, logical_count, num_local_physical_experts
        )
        physical_to_logical = np.take_along_axis(physical_to_logical, indices, axis=-1)

        # old slot -> new slot
        old_to_new = np.argsort(indices, axis=-1)
        mask = logical_to_physical == -1
        flat = np.where(mask, 0, logical_to_physical).reshape(num_moe_layers, -1).astype(np.int64)
        remapped = np.take_along_axis(old_to_new, flat, axis=-1)
        logical_to_physical = remapped.reshape(logical_to_physical.shape).astype(np.int32)
        logical_to_physical[mask] = -1

    return physical_to_logical, logical_to_physical, logical_count


def rank_loads(
    logical_tokens: np.ndarray,      # [layers, logical] or [steps, layers, logical]
    physical_to_logical_map: np.ndarray,
    logical_count: np.ndarray,
    capacity: int,
) -> np.ndarray:
    """Per-GPU load, using the upstream accounting convention.

    A replica carries `logical tokens / replica count`, the same assumption as
    upstream's `_log_rearrangement_metrics`. Returns [layers, ranks] for 2-D
    input and [steps, layers, ranks] for 3-D input.
    """
    per_copy = logical_tokens / logical_count
    shape = per_copy.shape
    if per_copy.ndim == 2:
        physical = np.take_along_axis(per_copy, physical_to_logical_map, axis=-1)  # [layers, physical]
        return physical.reshape(shape[0], -1, capacity).sum(axis=-1)
    if per_copy.ndim == 3:
        steps, layers, _ = per_copy.shape
        expanded = np.broadcast_to(physical_to_logical_map[None], (steps, layers, physical_to_logical_map.shape[1]))
        physical = np.take_along_axis(per_copy, expanded, axis=-1)                  # [steps, layers, physical]
        return physical.reshape(steps, layers, -1, capacity).sum(axis=-1)
    raise ValueError("logical_tokens must be [layers, logical] or [steps, layers, logical]")
