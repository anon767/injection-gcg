"""GPU attack runner (Kaggle). Attacks smolaya-guard (headline), protectai (one example), and
smolaya zero-shot, from HF. Writes results to OUT after EVERY attack so nothing can be lost.

Per detector: calibrate an operating threshold tau at 1% FPR on clean held-out benign, then run GCG
(early-stop at tau, full trajectory kept) on injections it flags. Success = decoded string scores
below tau. Held-out injections are deduped against smolaya-guard's training data (safe-guard+deepset).
"""
import json
import os
import re
import time

import numpy as np
import torch
from datasets import load_dataset

import detectors
import gcg
import universal

OUT = os.environ.get("ATTACK_OUT", "/kaggle/working/attack_results.json")
rng = np.random.default_rng(7)
clean = lambda t: re.sub(r"\s+", " ", str(t)).strip()
# smolaya-guard already done in a prior run; this run does the rest. protectai = one example.
PLAN = [("protectai", 1, dict(n_suffix=30, steps=350, batch=256, topk=256)),
        ("smolaya",   6, dict(n_suffix=30, steps=350, batch=256, topk=256))]


def pools(n_eval=60, n_train=24, n_benign=1500):
    guard_train, benign = set(), []
    for name in ["xTRam1/safe-guard-prompt-injection", "deepset/prompt-injections"]:
        d = load_dataset(name)
        for split in d:
            for t, y in zip(d[split]["text"], d[split]["label"]):
                c = clean(t); guard_train.add(c)
                if y == 0 and 10 < len(c) < 300:
                    benign.append(c)
    jv = load_dataset("jayavibhav/prompt-injection", split="test")
    inj = []
    for t, y in zip(jv["text"], jv["label"]):
        c = clean(t)
        if y == 1 and 10 < len(c) < 300 and c not in guard_train:
            inj.append(c)
    rng.shuffle(inj); rng.shuffle(benign)
    return {"eval_inj": inj[:n_eval], "train_inj": inj[n_eval:n_eval + n_train], "benign": benign[:n_benign]}


def ci(bools, B=5000):
    a = np.array(bools, float)
    if len(a) == 0:
        return [0.0, 0.0]
    bs = [a[rng.integers(0, len(a), len(a))].mean() for _ in range(B)]
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def asr_curve(attacks, tau, steps):
    out = []
    for k in range(steps):
        c = [(a["trajectory"][min(k, len(a["trajectory"]) - 1)] < tau) for a in attacks]
        out.append(float(np.mean(c)) if c else 0.0)
    return out


def save(out):
    json.dump(out, open(OUT, "w"), indent=1)


def main():
    P = pools()
    out = {"device": "cuda" if torch.cuda.is_available() else "cpu",
           "pools": {k: len(v) for k, v in P.items()}, "detectors": {}}
    save(out)
    for name, cap, budget in PLAN:
        det = detectors.load(name)
        tau = detectors.calibrate_tau(det, P["benign"], target_fpr=0.01)
        base = det.margin(P["eval_inj"]).numpy()
        flagged = [t for t, m in zip(P["eval_inj"], base) if m > tau][:cap]
        rec = {"tau": float(tau), "baseline_detected_at_tau": float((base > tau).mean()),
               "baseline_mean_margin": float(base.mean()), "n_flagged": len(flagged),
               "steps": budget["steps"], "attacks": []}
        out["detectors"][name] = rec
        save(out)
        t0 = time.time()
        for i, text in enumerate(flagged):
            r = gcg.attack(det, text, tau=tau, early_stop=True, seed=i, log=lambda *a: None, **budget)
            rec["attacks"].append(r)
            save(out)                      # <-- durable after EVERY attack
            print(f"[{name}] {i+1}/{len(flagged)} bypass={r['bypass']} "
                  f"{r['start_margin']:+.2f}->{r['string_margin']:+.2f} tau={tau:+.2f} "
                  f"steps={r['steps_run']} ({(time.time()-t0)/(i+1):.0f}s/atk)", flush=True)
        bys = [a["bypass"] for a in rec["attacks"]]
        rec["asr"] = float(np.mean(bys)) if bys else 0.0
        rec["asr_ci"] = ci(bys)
        rec["asr_curve"] = asr_curve(rec["attacks"], tau, budget["steps"])
        save(out)
        print(f"[{name}] DONE tau={tau:+.3f} recall@tau={rec['baseline_detected_at_tau']:.2f} "
              f"ASR={rec['asr']:.2f} {rec['asr_ci']}", flush=True)
        # universal suffix for the two real detectors (never let this abort the saved per-prompt data)
        if name in ("smolaya-guard", "protectai") and len(P["train_inj"]) >= 8:
            try:
                u = universal.universal(det, P["train_inj"][:20], n_suffix=30, steps=80, batch=192, log=print)
                m = det.margin([t + u["suffix"] for t in P["eval_inj"]]).numpy()
                u.update(eval_bypass=float((m < tau).mean()), eval_bypass_ci=ci((m < tau).tolist()),
                         eval_mean_margin=float(m.mean()), tau=float(tau))
                rec["universal"] = u
                save(out)
                print(f"[univ {name}] held-out bypass {u['eval_bypass']:.2f}", flush=True)
            except Exception as e:
                print(f"[univ {name}] FAILED: {e}", flush=True)
        del det
    # transfer across the real detectors
    out["transfer"] = {}
    names = [n for n, _, _ in PLAN if out["detectors"].get(n, {}).get("attacks")]
    for src in names:
        for dst in names:
            if src == dst:
                continue
            try:
                det = detectors.load(dst); tau = out["detectors"][dst]["tau"]
                strs = [a["text"] + a["suffix"] for a in out["detectors"][src]["attacks"] if a["bypass"]]
                if not strs:
                    continue
                m = det.margin(strs).numpy()
                out["transfer"][f"{src}->{dst}"] = {"n": len(strs), "bypass_on_dst": float((m < tau).mean()),
                                                    "ci": ci((m < tau).tolist()), "mean_margin": float(m.mean())}
                save(out); del det
                print(f"[transfer {src}->{dst}] {(m<tau).mean():.2f} of {len(strs)}", flush=True)
            except Exception as e:
                print(f"[transfer {src}->{dst}] FAILED: {e}", flush=True)
    save(out)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
