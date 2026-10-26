# -*- coding: utf-8 -*-
"""Run every experiment and write results/*.json plus the raw curves."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eplb.simulator import EPLBConfig, run, summarize  # noqa: E402
from eplb.traffic import traffic_stream  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

E = 384
LAYERS = 4
STEPS = 24000
BATCH = 512
TOPK = 8


def make_stream(**kw):
    base = dict(num_steps=STEPS, num_layers=LAYERS, num_experts=E,
                experts_per_token=TOPK, batch=BATCH, seed=42,
                concentration=1.0, skew=1.6, drift_period=6000,
                drift_strength=0.8, hot_fraction=0.05)
    base.update(kw)
    return traffic_stream(**base)


def exp_baselines():
    """EP8 vs EP64, with and without EPLB."""
    print("\n[exp] baselines")
    rows = {}
    for ep in (8, 64):
        for mode, extra in (("static", 0), ("static", 1), ("eplb", 1)):
            cfg = EPLBConfig(num_layers=LAYERS, num_experts=E, ep_size=ep,
                             extra_per_gpu=extra, window=500, interval=2000)
            t0 = time.time()
            res = run(make_stream(), cfg, mode=mode)
            stats = summarize(res["moe_lb"], warmup=2500)
            key = f"EP{ep}-{mode}-extra{extra}"
            rows[key] = {"ep": ep, "mode": mode, "extra": extra,
                         "stats": stats, "rearrangements": res["rearrangements"],
                         "seconds": round(time.time() - t0, 2)}
            np.save(OUT / f"lb_{key}.npy", res["moe_lb"])
            print(f"  {key:24} mean={stats['mean']:.4f} p95={stats['p95']:.4f} "
                  f"max={stats['max']:.3f} rearr={res['rearrangements']} ({time.time()-t0:.1f}s)")
    return rows


def exp_interval():
    """Sweep the rearrangement interval with the window held at 500."""
    print("\n[exp] interval sweep")
    out = []
    for interval in (500, 1000, 2000, 4000, 8000):
        cfg = EPLBConfig(num_layers=2, num_experts=E, ep_size=64, extra_per_gpu=1,
                         window=500, interval=interval)
        res = run(make_stream(num_layers=2, num_steps=16000), cfg, mode="eplb")
        stats = summarize(res["moe_lb"], warmup=2000)
        out.append({"interval": interval, "window": 500, **stats,
                    "rearrangements": res["rearrangements"]})
        print(f"  interval={interval:5} mean={stats['mean']:.4f} p95={stats['p95']:.4f} "
              f"rearr={res['rearrangements']:3}")
    return out


def exp_extra():
    """Sweep redundant experts. extra=0 still rearranges, isolating placement from replication."""
    print("\n[exp] extra-expert sweep")
    out = []
    for extra in (0, 1, 2, 3, 4, 6, 8):
        cfg = EPLBConfig(num_layers=2, num_experts=E, ep_size=64, extra_per_gpu=extra,
                         window=500, interval=2000)
        res = run(make_stream(num_layers=2, num_steps=16000), cfg, mode="eplb")
        stats = summarize(res["moe_lb"], warmup=2000)
        out.append({"extra_per_gpu": extra, "num_extra": res["num_extra"],
                    "capacity": res["capacity"], **stats})
        print(f"  extra={extra}/gpu physical={res['num_physical']:3} capacity={res['capacity']:2} "
              f"mean={stats['mean']:.4f} p95={stats['p95']:.4f}")
    return out


def exp_drift():
    """Recovery after the hot set migrates, for several rearrangement intervals."""
    print("\n[exp] drift recovery")
    rows = []
    for interval in (250, 1000, 4000):
        cfg = EPLBConfig(num_layers=LAYERS, num_experts=E, ep_size=64, extra_per_gpu=1,
                         window=min(500, interval), interval=interval)
        res = run(make_stream(num_steps=8000, drift_period=3000), cfg, mode="eplb")
        rows.append({"interval": interval, "lb": res["moe_lb"]})
        np.save(OUT / f"drift_{interval}.npy", res["moe_lb"])
        print(f"  interval={interval:5} post-shift mean(3000-8000)="
              f"{res['moe_lb'][3000:].mean():.4f}")

    cfg = EPLBConfig(num_layers=LAYERS, num_experts=E, ep_size=64, extra_per_gpu=0,
                     window=500, interval=0)
    res = run(make_stream(num_steps=8000, drift_period=3000), cfg, mode="static")
    np.save(OUT / "drift_static.npy", res["moe_lb"])
    print(f"  static              post-shift mean={res['moe_lb'][3000:].mean():.4f}")
    return rows


def main():
    t0 = time.time()
    results = {
        "config": {"experts": E, "layers": LAYERS, "steps": STEPS,
                   "batch": BATCH, "top_k": TOPK, "seed": 42},
        "baselines": exp_baselines(),
        "interval": exp_interval(),
        "extra": exp_extra(),
        "drift_intervals": [250, 1000, 4000],
    }
    exp_drift()
    (OUT / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nall done in {time.time()-t0:.1f}s -> {OUT/'results.json'}")


if __name__ == "__main__":
    main()
