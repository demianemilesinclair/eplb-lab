# eplb-lab

A NumPy reproduction of the expert-parallel load balancer (EPLB) used in
[DeepInfer](https://github.com/deepseek-ai/DeepInfer). It runs on CPU and needs
no GPU.

The simulation tests whether placement alone is enough, how many redundant
experts are needed, and how the rearrangement interval affects load balance.

The full suite covers 24 configurations and about 40,000 simulated steps, and
runs in under three minutes.

## Quick start

```bash
python -m pip install numpy
python experiments/run_all.py      # experiments, writes results/
python experiments/make_figs.py    # figures, writes figs/
python tests/test_policy.py        # correctness checks
```

The longer write-up is in [eplb-reproduction.md](eplb-reproduction.md).

## Results

Measurements use a 384-expert, 4-layer model with 512 tokens per step (top-8),
and a synthetic traffic trace that changes its hot expert set every 6000 steps.

| Configuration | Mean moe_lb | p95 |
|---|---|---|
| EP8, no EPLB | 1.32 | 1.63 |
| EP8 with EPLB | 1.04 | 1.11 |
| EP64, no EPLB | 3.90 | 7.46 |
| EP64 with EPLB | 1.15 | 1.42 |

`moe_lb` is the slowest rank's load divided by the mean rank load.

| Change | Effect on steady-state moe_lb |
|---|---|
| Placement only, no redundancy (EP64) | 4.08 |
| One redundant expert per GPU | 1.19 |
| Eight redundant experts per GPU | 1.12 |
| Rearrangement interval 500 vs 8000 steps | 1.176 vs 1.465 |

![Convergence and recovery](figs/fig_main.png)

![Interval and redundancy sweeps](figs/fig_sweep.png)

## Files

```
eplb/policy.py        NumPy port of DeepInfer's policy.py
eplb/simulator.py     Simulation loop, matching upstream ExpertLoadBalancer timing
eplb/traffic.py       Synthetic MoE traffic with static skew and periodic drift
tests/test_policy.py  Invariants, random-placement comparison, brute-force check
experiments/          Experiment runner and plotting
results/              Per-step curves and the summary JSON
```

## Relationship to upstream

| This repo | DeepInfer source | Notes |
|---|---|---|
| `resolve_num_extra_experts` | `eplb/__init__.py`, same name | Rounds the redundancy count to a multiple of EP |
| `rebalance_experts_global` | `eplb/policy.py`, same name | Replication and placement |
| `_batch_greedy_indices` | `eplb/policy.py`, same name | Objective: `min sum over steps of max rank load` |
| `build_state`, `run` | `ExpertLoadBalancer` | Ring buffer; first rearrangement at one quarter of the interval |
| `rank_loads` | `_log_rearrangement_metrics` | One replica carries logical load divided by replica count |

Two upstream details are preserved because they change behaviour:

- the `+ arange * 1e-4` bias in the replication greedy. It only breaks ties
  between experts with equal load;
- the first rearrangement happening at a quarter of the interval, which reduces
  the cold-start imbalance.

## Verification

`tests/test_policy.py` checks four properties:

1. Every logical expert has at least one replica, every GPU holds exactly
   `capacity` slots, and the two mappings are inverses.
2. Random placement never beats the greedy, over 200 trials.
3. Uniform load is balanced exactly.
4. Against an exhaustive search on a 4-expert, 6-slot instance, the greedy is
   0.40% off optimal on average and 2.37% in the worst case.

## Limitations

- The traffic is synthetic. Absolute numbers are not comparable to the 1.09
  reported for production; only the relative behaviour of the strategies carries
  over.
- No all-to-all communication, weight shuffling, or GPU timing is modelled.
- Layers are balanced independently, while the metric sums across layers per GPU,
  which matches the upstream convention.

## License

`eplb/policy.py` is a port of a DeepInfer algorithm under Apache-2.0; the scope
is described in [THIRD_PARTY.md](THIRD_PARTY.md). Everything else is MIT, see
[LICENSE](LICENSE).
