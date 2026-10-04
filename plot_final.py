"""Figures for the blog, from results/final.json. Styled to match the existing posts."""
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
NICE = {"protectai": "protectai/deberta-v3\n(purpose-built)", "smolaya-guard": "smolaya-guard\n(ours, fine-tuned)",
        "smolaya": "smolaya\n(zero-shot)"}
exp = json.load(open("results/final.json"))
D = exp["detectors"]


def fig_margins():
    dets = list(D)
    fig, axes = plt.subplots(1, len(dets), figsize=(4.8 * len(dets), 4.3), squeeze=False)
    for ax, n in zip(axes[0], dets):
        A = D[n]["attacks"]; tau = D[n]["tau"]
        b = np.array([a["start_margin"] for a in A]); a2 = np.array([a["string_margin"] for a in A])
        for i in range(len(b)):
            ax.plot([0, 1], [b[i], a2[i]], color=(HIT if a2[i] < tau else MISS), alpha=.55, lw=1)
        ax.scatter(np.zeros_like(b), b, s=20, color=INK, zorder=3)
        ax.scatter(np.ones_like(a2), a2, s=20, color=[(HIT if x < tau else MISS) for x in a2], zorder=3)
        ax.axhline(tau, color=ACC, lw=1.2, ls="--")
        ax.text(1.28, tau, "  τ (1% FPR)", color=ACC, va="center", fontsize=9)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["original\ninjection", "+ adversarial\nsuffix"])
        ax.set_xlim(-.3, 1.5)
        ax.set_title(f"{NICE.get(n, n)}\nASR {D[n]['asr']*100:.0f}%")
        ax.set_ylabel("injection margin  (logit injection − benign)")
    fig.tight_layout(); fig.savefig(f"{OUT}/gcg-margins.png", bbox_inches="tight"); plt.close(fig)


def fig_budget():
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for n, col in zip(D, (HIT, ACC, MISS)):
        c = D[n].get("asr_curve")
        if c:
            ax.plot(range(1, len(c) + 1), np.array(c) * 100, lw=2, color=col, label=NICE.get(n, n).replace("\n", " "))
    ax.set_xlabel("GCG steps (attacker budget)"); ax.set_ylabel("attack success rate (%)")
    ax.set_ylim(0, 103); ax.legend(frameon=False, fontsize=10)
    ax.set_title("A stronger detector just buys more iterations, not safety")
    fig.tight_layout(); fig.savefig(f"{OUT}/gcg-budget.png", bbox_inches="tight"); plt.close(fig)


def fig_summary():
    labels, vals, los, his, cols = [], [], [], [], []
    def add(lbl, v, ci, col):
        labels.append(lbl); vals.append(v); los.append(v - ci[0]); his.append(ci[1] - v); cols.append(col)
    for n in D:
        add(f"per-prompt\n{n}", D[n]["asr"], D[n]["asr_ci"], HIT)
    for n in D:
        u = D[n].get("universal")
        if u:
            add(f"universal\n{n}", u["eval_bypass"], u["eval_bypass_ci"], ACC)
    for k, v in exp.get("transfer", {}).items():
        add(f"transfer\n{k}", v["bypass_on_dst"], v["ci"], MISS)
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(1.5 * len(labels) + 2, 4.4))
    ax.bar(x, vals, color=cols, yerr=[los, his], capsize=4, width=.62)
    for xi, v in zip(x, vals):
        ax.text(xi, min(v + .03, 1.02), f"{v*100:.0f}%", ha="center", fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.1); ax.set_ylabel("bypass rate (vs τ at 1% FPR)")
    ax.set_title("Per-prompt, one universal suffix, and cross-model transfer (95% CI)")
    fig.tight_layout(); fig.savefig(f"{OUT}/gcg-summary.png", bbox_inches="tight"); plt.close(fig)


fig_margins(); fig_budget(); fig_summary()
print("wrote figures to", OUT)
