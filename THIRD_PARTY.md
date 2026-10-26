# Third-party notices

## DeepSeek-AI/DeepInfer

`eplb/policy.py` is a port of the EPLB policy from
[DeepSeek-AI/DeepInfer](https://github.com/deepseek-ai/DeepInfer),
which is licensed under the Apache License, Version 2.0.

What was ported:

- `resolve_num_extra_experts`: identical logic (redundancy count rounded up to a multiple of `ep_size`)
- `rebalance_experts_global`: identical target function and greedy ordering
  (redundancy allocation minimising the max per-copy load; placement minimising
  `sum over steps of max over ranks`)
- `_batch_greedy_indices`: identical objective and slot-occupancy rule

What changed: PyTorch tensor operations were re-expressed with NumPy so the
algorithm can run on CPU-only machines. No target function, ordering rule, or
tie-breaking behaviour was altered, including the `+ arange * 1e-4` bias used
to break ties in the redundancy allocation.

The rest of this repository (simulator, traffic model, experiments, tests and
the article) is original work licensed under the MIT License (see `LICENSE`).

A copy of the Apache License 2.0 is provided in
[`licenses/APACHE-2.0.txt`](licenses/APACHE-2.0.txt).
