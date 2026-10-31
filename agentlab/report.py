"""Figures for the README from results/study.json.

    python -m agentlab.report
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RES = Path("results")
INK, MUTED = "#1c1c1a", "#62625c"
S1, S2 = "#2a78d6", "#eb6834"


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8.5)


def fig_controls(d):
    none = d["matrix"]["no controls | planner with mistakes (30%)"]["by_category"]
    gov = d["matrix"]["governed | planner with mistakes (30%)"]["by_category"]
    cats = sorted(none, key=lambda c: -none[c]["unsafe"] / none[c]["runs"])
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    h = 0.38
    for i, c in enumerate(cats):
        a = 100 * none[c]["unsafe"] / none[c]["runs"]
        b = 100 * gov[c]["unsafe"] / gov[c]["runs"]
        ax.barh(i - h / 2, a, h, color=S2, label="No runtime controls" if i == 0 else None)
        ax.barh(i + h / 2, max(b, 0.4), h, color=S1, label="Governed runtime" if i == 0 else None)
        ax.text(a + 1.5, i - h / 2, f"{a:.0f}%", va="center", fontsize=8, color=INK)
        ax.text(max(b, 0.4) + 1.5, i + h / 2, f"{b:.0f}%", va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(cats)))
    ax.set_yticklabels(cats, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 110)
    ax.set_xlabel("Runs ending in an unsafe outcome (%), planner making mistakes in 30% of opportunities",
                  color=MUTED, fontsize=8.5)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("Same planner, same requests: what the runtime controls prevent", loc="left", fontsize=10.5, color=INK)
    style(ax)
    fig.tight_layout()
    fig.savefig(RES / "controls.png", dpi=160)


def fig_crash(d):
    rows = list(d["crash"].items())
    fig, ax = plt.subplots(figsize=(7.2, 2.4))
    for i, (k, v) in enumerate(rows):
        n = v["duplicate_effects"]
        ax.barh(i, max(n, 0.15), color=S2 if n else S1, height=0.55)
        ax.text(max(n, 0.15) + 0.6, i, f"{n} of {v['cases']}", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 42)
    ax.set_xlabel("Crash-and-resume cases that repeated a side effect (refund, SMS, assignment, claim)",
                  color=MUTED, fontsize=8.5)
    ax.set_title("Killing the worker mid-write: what it takes to never repeat an action", loc="left",
                 fontsize=10.5, color=INK)
    style(ax)
    fig.tight_layout()
    fig.savefig(RES / "crash.png", dpi=160)


def main():
    d = json.loads((RES / "study.json").read_text())
    fig_controls(d)
    fig_crash(d)
    print("figures written to", RES)


if __name__ == "__main__":
    main()
