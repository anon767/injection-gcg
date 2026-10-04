"""Figures for the blog post, styled to match the existing posts (clean matplotlib, muted palette)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.expanduser("~/blogdraft/images")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "sans-serif", "font.size": 12, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 128, "axes.grid": True,
                     "grid.alpha": 0.25, "grid.linewidth": 0.6})
INK, HIT, MISS, ACC = "#1a1a1a", "#c0392b", "#7f8c8d", "#2c7fb8"

exp = json.load(open("results/experiment.json"))


def fig_margins():
    """Per-prompt: margin before vs after attack, one detector per panel."""
    dets = list(exp["per_prompt"])
    fig, axes = plt.subplots(1, len(dets), figsize=(5.2 * len(dets), 4.2))
    if len(dets) == 1:
        axes = [axes]
    for ax, n in zip(axes, dets):
        A = exp["per_prompt"][n]["attacks"]
        b = np.array([a["start_margin"] for a in A])
        a = np.array([a["string_margin"] for a in A])
        for i in range(len(b)):
            ax.plot([0, 1], [b[i], a[i]], color=(HIT if a[i] < 0 else MISS), alpha=.5, lw=1)
        ax.scatter(np.zeros_like(b), b, s=18, color=INK, zorder=3)
        ax.scatter(np.ones_like(a), a, s=18, color=[(HIT if x < 0 else MISS) for x in a], zorder=3)
        ax.axhline(0, color=INK, lw=1, ls="--")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["original\ninjection", "+ adversarial\nsuffix"])
        ax.set_xlim(-.3, 1.3)
        asr = exp["per_prompt"][n]["asr"]
        ax.set_title(f"{n}\nASR {asr*100:.0f}%  (below 0 = scored benign)")
        ax.set_ylabel("injection margin  (logit injection − benign)")
    fig.tight_layout(); fig.savefig(f"{OUT}/gcg-margins.png", bbox_inches="tight"); plt.close(fig)


def fig_summary():
    """ASR per detector + universal + transfer, as a labelled bar chart with CIs."""
    labels, vals, los, his, cols = [], [], [], [], []
    for n in exp["per_prompt"]:
        labels.append(f"per-prompt\n{n}"); vals.append(exp["per_prompt"][n]["asr"])
        lo, hi = exp["per_prompt"][n]["asr_ci"]; los.append(vals[-1] - lo); his.append(hi - vals[-1])
        cols.append(HIT)
    for n in exp.get("universal", {}):
        labels.append(f"universal\n{n}"); vals.append(exp["universal"][n]["eval_bypass"])
        lo, hi = exp["universal"][n]["eval_bypass_ci"]; los.append(vals[-1] - lo); his.append(hi - vals[-1])
        cols.append(ACC)
    for k, v in exp.get("transfer", {}).items():
        labels.append(f"transfer\n{k}"); vals.append(v["bypass_on_dst"])
        lo, hi = v["ci"]; los.append(vals[-1] - lo); his.append(hi - vals[-1]); cols.append(MISS)
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(1.7 * len(labels) + 2, 4.4))
    ax.bar(x, vals, color=cols, yerr=[los, his], capsize=4, width=.6)
    for xi, v in zip(x, vals):
        ax.text(xi, v + .02, f"{v*100:.0f}%", ha="center", fontsize=11)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.08); ax.set_ylabel("bypass rate")
    ax.set_title("Fraction of injections scored benign after a white-box suffix (95% CI)")
    fig.tight_layout(); fig.savefig(f"{OUT}/gcg-summary.png", bbox_inches="tight"); plt.close(fig)


fig_margins()
fig_summary()
print("wrote", OUT)
