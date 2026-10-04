"""Final attack experiment for the blog.

For each detector: calibrate an operating threshold tau at FPR=1% on held-out benign text, then run
GCG (real budget, no early stop so we get the full attack-success-vs-budget curve) on injections the
detector flags at tau. Reports ASR@tau, the budget curve, a universal suffix, and cross-model transfer.

Held-out set: jayavibhav/prompt-injection TEST, deduplicated against everything smolaya-guard trained
on (safe-guard + deepset, all splits), so attacking the shipped guard model is leakage-free and the
same set is used for every detector (so transfer is comparable).
"""
import json
import re
import sys
import time

import numpy as np
from datasets import load_dataset

import detectors
import gcg
import universal

rng = np.random.default_rng(7)
clean = lambda t: re.sub(r"\s+", " ", str(t)).strip()


def pools(n_eval=40, n_train=30, n_benign=1500):
    guard_train = set()
    for name in ["xTRam1/safe-guard-prompt-injection", "deepset/prompt-injections"]:
        d = load_dataset(name)
        for split in d:
            guard_train.update(clean(t) for t in d[split]["text"])
    jv = load_dataset("jayavibhav/prompt-injection", split="test")
    inj, ben = [], []
    for t, y in zip(jv["text"], jv["label"]):
        c = clean(t)
        if not (10 < len(c) < 400) or c in guard_train:
            continue
        (inj if y == 1 else ben).append(c)
    rng.shuffle(inj); rng.shuffle(ben)
    return {"eval_inj": inj[:n_eval], "train_inj": inj[n_eval:n_eval + n_train], "benign": ben[:n_benign]}


def ci(bools, B=5000):
    a = np.array(bools, float)
    if len(a) == 0:
        return [0.0, 0.0]
    bs = [a[rng.integers(0, len(a), len(a))].mean() for _ in range(B)]
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def asr_curve(attacks, tau, steps):
    """fraction whose best margin is below tau by step k, for k in 1..steps (string-verified at end)."""
    curve = []
    for k in range(steps):
        c = [(a["trajectory"][min(k, len(a["trajectory"]) - 1)] < tau) for a in attacks]
        curve.append(float(np.mean(c)) if c else 0.0)
    return curve


def main():
    names = sys.argv[1:] or ["protectai", "smolaya-guard"]
    P = pools()
    out = {"pools": {k: len(v) for k, v in P.items()}, "detectors": {}}
    import os
    if os.path.exists("results/final.json"):          # merge, so separate runs accumulate
        prev = json.load(open("results/final.json"))
        out["detectors"].update(prev.get("detectors", {}))
    # CPU-sized: smaller batch, more steps => same total candidate exploration, full budget curve.
    budget = {"protectai": dict(n_suffix=30, steps=250, batch=48, topk=256),
              "smolaya-guard": dict(n_suffix=30, steps=200, batch=32, topk=256),
              "smolaya": dict(n_suffix=30, steps=200, batch=32, topk=256)}
    CAP = 12  # attacked prompts per detector

    for n in names:
        det = detectors.load(n)
        tau = detectors.calibrate_tau(det, P["benign"], target_fpr=0.01)
        base = det.margin(P["eval_inj"]).numpy()
        flagged = [t for t, m in zip(P["eval_inj"], base) if m > tau][:CAP]
        rec = {"tau": tau, "baseline_detected_at_tau": float((base > tau).mean()),
               "baseline_mean_margin": float(base.mean()), "n_flagged": len(flagged), "attacks": []}
        b = budget[n]
        t0 = time.time()
        for i, text in enumerate(flagged):
            r = gcg.attack(det, text, tau=tau, early_stop=False, seed=i, log=lambda *a: None, **b)
            rec["attacks"].append(r)
            print(f"[{n}] {i+1}/{len(flagged)} bypass={r['bypass']} "
                  f"{r['start_margin']:+.2f}->{r['string_margin']:+.2f} tau={tau:+.2f} "
                  f"({(time.time()-t0)/(i+1):.0f}s/atk)", flush=True)
        bys = [a["bypass"] for a in rec["attacks"]]
        rec["asr"] = float(np.mean(bys)) if bys else 0.0
        rec["asr_ci"] = ci(bys)
        rec["asr_curve"] = asr_curve(rec["attacks"], tau, b["steps"])
        out["detectors"][n] = rec
        json.dump(out, open("results/final.json", "w"), indent=1)
        print(f"[{n}] tau={tau:+.3f} baseline-recall@tau={rec['baseline_detected_at_tau']:.2f} "
              f"ASR={rec['asr']:.2f} {rec['asr_ci']}", flush=True)

        u = universal.universal(det, P["train_inj"][:20], n_suffix=30, steps=150, batch=48, log=print)
        m = det.margin([t + u["suffix"] for t in P["eval_inj"]]).numpy()
        u.update(eval_bypass=float((m < tau).mean()), eval_bypass_ci=ci((m < tau).tolist()),
                 eval_mean_margin=float(m.mean()), tau=tau)
        rec["universal"] = u
        out["detectors"][n] = rec
        json.dump(out, open("results/final.json", "w"), indent=1)
        print(f"[univ {n}] held-out bypass {u['eval_bypass']:.2f}", flush=True)

    # transfer: per-prompt suffixes that bypassed src, replayed as strings against dst, vs dst's tau
    out["transfer"] = {}
    allnames = list(out["detectors"])
    for src in allnames:
        for dst in allnames:
            if src == dst or not out["detectors"][src].get("attacks"):
                continue
            det = detectors.load(dst); tau = out["detectors"][dst]["tau"]
            strs = [a["text"] + a["suffix"] for a in out["detectors"][src]["attacks"] if a["bypass"]]
            if not strs:
                continue
            m = det.margin(strs).numpy()
            out["transfer"][f"{src}->{dst}"] = {"n": len(strs), "bypass_on_dst": float((m < tau).mean()),
                                                "ci": ci((m < tau).tolist()), "mean_margin": float(m.mean())}
            print(f"[transfer {src}->{dst}] {(m<tau).mean():.2f} of {len(strs)}", flush=True)
    json.dump(out, open("results/final.json", "w"), indent=1)
    print("done")


if __name__ == "__main__":
    main()
