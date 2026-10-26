# -*- coding: utf-8 -*-
"""Render the experiment figures from results/."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "figs"
FIG.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.unicode_minus": False,
    "figure.dpi": 160,
    "savefig.dpi": 160,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#9CA3AF",
    "axes.labelcolor": "#1C1E24",
    "text.color": "#1C1E24",
    "xtick.color": "#4B5563",
    "ytick.color": "#4B5563",
})

C = {
    "static": "#9CA3AF",
    "ep8": "#F59E0B",
    "ep64": "#2563EB",
    "good": "#10B981",
    "bad": "#DC2626",
}


def smooth(x, k=50):
    if k <= 1:
        return x
    kernel = np.ones(k) / k
    pad = k // 2
    padded = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[:len(x)]


def fig_main():
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.6))

    # ---- (a) convergence ----
    ax = axes[0]
    series = [
        ("EP64-eplb-extra1", "EP64 + EPLB", C["ep64"], 2.0),
        ("EP64-static-extra1", "EP64, no EPLB", C["ep64"], 1.8),
        ("EP8-eplb-extra1", "EP8 + EPLB", C["ep8"], 2.0),
        ("EP8-static-extra1", "EP8, no EPLB", C["ep8"], 1.8),
    ]
    for key, label, color, lw in series:
        lb = np.load(RES / f"lb_{key}.npy")
        ls = "-" if "eplb" in key else "--"
        ax.plot(np.arange(len(lb)), smooth(lb, 60), label=label, color=color, lw=lw, ls=ls, alpha=0.95)

    ax.set_xlabel("decode step")
    ax.set_ylabel("moe_lb (slowest GPU / mean)")
    ax.set_title("(a) Load imbalance without EPLB, by EP size", fontsize=12, loc="left", pad=10)
    ax.set_ylim(0.9, 8.6)
    ax.axhline(1.0, color="#D1D5DB", lw=1, ls=":")
    ax.legend(frameon=False, fontsize=9.5, loc="upper center", ncol=2)
    ax.grid(axis="y", color="#EEF0F3", lw=0.8)
    ax.set_axisbelow(True)
    ax.annotate("peak 7.3x", xy=(14900, 7.2), xytext=(10600, 6.55), fontsize=10.5,
                color=C["bad"], fontweight="bold")

    # ---- (b) drift recovery ----
    ax = axes[1]
    shift = 3000
    end = 5900
    for interval, color, label in ((250, "#059669", "interval 250"),
                                   (1000, "#2563EB", "interval 1000"),
                                   (4000, "#F59E0B", "interval 4000")):
        lb = np.load(RES / f"drift_{interval}.npy")[:end]
        ax.plot(np.arange(len(lb)), smooth(lb, 30), color=color, lw=1.9, label=label)
    lb = np.load(RES / "drift_static.npy")[:end]
    ax.plot(np.arange(len(lb)), smooth(lb, 30), color=C["static"], lw=1.8, ls="--", label="no EPLB")

    ax.axvline(shift, color=C["bad"], lw=1.4, ls="-", alpha=0.8)
    ax.annotate("hot set migrates", xy=(shift, 4.4), xytext=(shift + 180, 4.55),
                fontsize=10, color=C["bad"])
    ax.annotate("interval 4000: waits 1000 steps", xy=(3620, 1.44), xytext=(3080, 2.05),
                fontsize=9, color="#B45309",
                arrowprops=dict(arrowstyle="->", color="#B45309", lw=1))
    ax.set_xlabel("decode step")
    ax.set_ylabel("moe_lb")
    ax.set_title("(b) Recovery after the hot set changes, by rearrangement interval",
                 fontsize=12, loc="left", pad=10)
    ax.set_ylim(0.95, 5.2)
    ax.axhline(1.0, color="#D1D5DB", lw=1, ls=":")
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    ax.grid(axis="y", color="#EEF0F3", lw=0.8)
    ax.set_axisbelow(True)

    fig.tight_layout()
    out = FIG / "fig_main.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", out)


def fig_sweep():
    data = json.loads((RES / "results.json").read_text(encoding="utf-8"))

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.4))

    # ---- (a) interval ----
    ax = axes[0]
    iv = data["interval"]
    x = [d["interval"] for d in iv]
    y = [d["mean"] for d in iv]
    ax.plot(x, y, marker="o", ms=6, lw=2.2, color=C["ep64"], label="steady-state moe_lb")
    ax.fill_between(x, [d["p95"] for d in iv], y, color=C["ep64"], alpha=0.12, label="p95 tail")
    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.get_xaxis().set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
    ax.set_xlabel("Rearrangement interval (steps, window fixed at 500)")
    ax.set_ylabel("moe_lb (steady-state mean)")
    ax.set_title("(a) Load imbalance versus rearrangement interval", fontsize=12, loc="left", pad=10)
    ax.grid(axis="y", color="#EEF0F3", lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")

    ax2 = ax.twinx()
    ax2.bar(x, [d["rearrangements"] for d in iv], width=0.32 * np.array(x),
            color="#E5E7EB", zorder=0)
    ax2.set_ylabel("Rearrangements within 16,000 steps", color="#6B7280")
    ax2.tick_params(axis="y", colors="#6B7280")
    ax2.set_ylim(0, 40)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color("#D1D5DB")
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)

    # ---- (b) redundancy ----
    ax = axes[1]
    ex = data["extra"]
    x = [d["extra_per_gpu"] for d in ex]
    y = [d["mean"] for d in ex]
    ax.plot(x, y, marker="s", ms=6, lw=2.2, color=C["good"])
    ax.set_xlabel("Redundant experts per GPU (number above each point = slots per GPU)")
    ax.set_ylabel("moe_lb (steady-state mean)")
    ax.set_title("(b) Load imbalance versus redundant experts per GPU", fontsize=12, loc="left", pad=10)
    ax.grid(axis="y", color="#EEF0F3", lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlim(-0.5, 9.6)
    ax.set_ylim(0.95, 4.5)

    for d in ex:
        ax.annotate(f"{d['capacity']}", xy=(d["extra_per_gpu"], d["mean"]),
                    xytext=(0, 10), textcoords="offset points",
                    ha="center", fontsize=8.5, color="#6B7280")

    ax.annotate("Placement only, no replicas: 4.08\n(4.69 with no rearrangement at all)",
                xy=(0.06, 4.03), xytext=(3.6, 3.75),
                fontsize=9.2, color=C["bad"], va="center",
                arrowprops=dict(arrowstyle="->", color=C["bad"], lw=1))
    ax.annotate("One extra replica: 1.19 (published setting)", xy=(1.0, 1.20), xytext=(3.6, 2.55),
                fontsize=9.2, color="#047857", va="center",
                arrowprops=dict(arrowstyle="->", color="#047857", lw=1))
    ax.annotate("More replicas: 1.17 -> 1.12, flat", xy=(4.0, 1.14), xytext=(3.6, 1.55),
                fontsize=9.2, color="#047857", va="center",
                arrowprops=dict(arrowstyle="->", color="#047857", lw=1))

    fig.tight_layout()
    out = FIG / "fig_sweep.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    fig_main()
    fig_sweep()
