"""Full blog experiment:
  1. Per-prompt GCG attack on each detector over K held-out injections -> attack success rate (ASR) + CI.
  2. One universal suffix per detector, trained on a disjoint set, evaluated on the same K held-out.
  3. Cross-model transfer: suffixes found on detector A, replayed (as strings) against detector B.
All success is measured on re-tokenised strings (what an attacker sends)."""
import json
import sys
import time

import numpy as np
from datasets import load_dataset

import detectors
import gcg
import universal

rng = np.random.default_rng(1)


def injections(n_eval, n_train):
    d = load_dataset("xTRam1/safe-guard-prompt-injection", split="train")
    inj = [t for t, y in zip(d["text"], d["label"]) if y == 1 and 10 < len(t) < 400]
    rng.shuffle(inj)
    return inj[:n_eval], inj[n_eval:n_eval + n_train]


def ci(bools, B=5000):
    a = np.array(bools, float)
    bs = [a[rng.integers(0, len(a), len(a))].mean() for _ in range(B)]
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def main():
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    names = sys.argv[2:] or ["smolaya", "protectai"]
    eval_inj, train_inj = injections(K, 30)
    dets = {n: detectors.load(n) for n in names}
    out = {"k": K, "eval_inj": eval_inj, "per_prompt": {}, "universal": {}, "transfer": {}}

    # baseline: attack ONLY injections the detector actually flags (margin>0); bypassing an
    # already-benign-scored one is meaningless. Report both the base detection rate and the ASR.
    for n, det in dets.items():
        base = det.margin(eval_inj).numpy()
        flagged = [t for t, m in zip(eval_inj, base) if m > 0]
        out["per_prompt"][n] = {"baseline_detected": float((base > 0).mean()),
                                "baseline_mean_margin": float(base.mean()),
                                "n_flagged": len(flagged), "attacks": []}
        out["per_prompt"][n]["_flagged"] = flagged[:18]  # cap attacked prompts to bound CPU time

    # 1. per-prompt attacks (on flagged injections only)
    for n, det in dets.items():
        t0 = time.time()
        flagged = out["per_prompt"][n].pop("_flagged")
        for i, text in enumerate(flagged):
            r = gcg.attack(det, text, n_suffix=20, steps=40, topk=128, batch=64, log=lambda *a: None)
            out["per_prompt"][n]["attacks"].append(r)
            print(f"[{n}] {i+1}/{len(flagged)} bypass={r['bypass']} {r['start_margin']:+.2f}->{r['string_margin']:+.2f} "
                  f"({(time.time()-t0)/(i+1):.0f}s/atk)", flush=True)
        bys = [a["bypass"] for a in out["per_prompt"][n]["attacks"]]
        out["per_prompt"][n]["asr"] = float(np.mean(bys))
        out["per_prompt"][n]["asr_ci"] = ci(bys)
        json.dump(out, open("results/experiment.json", "w"), indent=1)

    # 2. universal suffix per detector
    for n, det in dets.items():
        u = universal.universal(det, train_inj, n_suffix=20, steps=80, batch=96, log=print)
        strs = [t + u["suffix"] for t in eval_inj]
        m = det.margin(strs).numpy()
        u["eval_bypass"] = float((m < 0).mean()); u["eval_bypass_ci"] = ci((m < 0).tolist())
        u["eval_mean_margin"] = float(m.mean())
        out["universal"][n] = u
        print(f"[univ {n}] bypass {u['eval_bypass']:.2f} on held-out, suffix={u['suffix']!r}", flush=True)
        json.dump(out, open("results/experiment.json", "w"), indent=1)

    # 3. transfer per-prompt suffixes across detectors (string replay)
    for src in names:
        for dst in names:
            if src == dst:
                continue
            det = dets[dst]
            strs = [a["text"] + a["suffix"] for a in out["per_prompt"][src]["attacks"] if a["bypass"]]
            if not strs:
                continue
            m = det.margin(strs).numpy()
            out["transfer"][f"{src}->{dst}"] = {"n": len(strs), "bypass_on_dst": float((m < 0).mean()),
                                                "ci": ci((m < 0).tolist()), "mean_margin": float(m.mean())}
            print(f"[transfer {src}->{dst}] {(m<0).mean():.2f} of {len(strs)} suffixes also bypass {dst}",
                  flush=True)
    json.dump(out, open("results/experiment.json", "w"), indent=1)
    print("done")


if __name__ == "__main__":
    main()
