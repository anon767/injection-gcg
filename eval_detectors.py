"""Baseline detection quality on held-out test splits, with bootstrap 95% CIs."""
import json
import sys
import time

import numpy as np
from datasets import load_dataset
from sklearn.metrics import roc_auc_score

import detectors

N = 400
rng = np.random.default_rng(0)


def sample(name, n=N):
    d = load_dataset(name, split="test")
    idx = rng.permutation(len(d))[: min(n, len(d))]
    d = d.select(idx)
    return list(d["text"]), np.array(d["label"])


def boot(y, s, fn, B=2000):
    vals = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) < 2:
            continue
        vals.append(fn(y[i], s[i]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


sets = {k: sample(k) for k in ["deepset/prompt-injections", "xTRam1/safe-guard-prompt-injection"]}
res = {}
for dname in sys.argv[1:] or ["smolaya", "protectai"]:
    det = detectors.load(dname)
    for sname, (X, y) in sets.items():
        t0 = time.time()
        s = det.margin(X).numpy()
        dt = (time.time() - t0) / len(X)
        tpr = lambda y, s: float((s[y == 1] > 0).mean())
        fpr = lambda y, s: float((s[y == 0] > 0).mean())
        r = {"n": len(y), "auc": float(roc_auc_score(y, s)), "auc_ci": boot(y, s, roc_auc_score),
             "tpr@0": tpr(y, s), "tpr_ci": boot(y, s, tpr), "fpr@0": fpr(y, s), "fpr_ci": boot(y, s, fpr),
             "sec_per_item": dt}
        res[f"{dname}|{sname}"] = r
        print(dname, sname, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}),
              flush=True)
json.dump(res, open(f"results/baseline_{'_'.join(sys.argv[1:]) or 'all'}.json", "w"), indent=1)
