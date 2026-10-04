"""Universal GCG suffix: optimise ONE suffix over many injections at once (Zou et al. 2023, sec 3.2),
then measure its bypass rate on held-out injections. This is the stronger claim: a single fixed string
an attacker can paste onto any injection, not a per-prompt re-optimisation."""
import random
import time

import torch

from gcg import allowed_tokens, roundtrip_ok, suffix_str


def _batch_margin(det, pairs):
    """pairs: list of (pre, suf, tail) id-lists -> margins [B], via embedding lookup (no grad)."""
    E_table = det.embed_matrix().detach()
    L = max(len(p) + len(s) + len(t) for p, s, t in pairs)
    Eb = torch.zeros(len(pairs), L, E_table.size(1)); ab = torch.zeros(len(pairs), L, dtype=torch.long)
    for b, (p, s, t) in enumerate(pairs):
        e = E_table[torch.tensor(p + s + t)]
        Eb[b, :e.size(0)] = e; ab[b, :e.size(0)] = 1
    mk = torch.tensor([det.markers] * len(pairs)) if hasattr(det, "markers") else None
    with torch.no_grad():
        return det.margin_from_embeds(Eb, ab, mk)


def universal(det, texts, n_suffix=20, steps=200, topk=256, batch=192, seed=0, log=print):
    rng = random.Random(seed)
    vocab = allowed_tokens(det.tok)
    E_table = det.embed_matrix().detach()
    tids = [det.tok(t.replace(getattr(det.tok, "mask_token", "\0"), " "),
                    add_special_tokens=False)["input_ids"] for t in texts]
    suf = vocab[torch.randint(0, len(vocab), (n_suffix,),
                              generator=torch.Generator().manual_seed(seed))].tolist()

    def splits(suffix):
        return [det.build(tid, suffix) for tid in tids]

    def mean_margin(suffix):
        return float(_batch_margin(det, splits(suffix)).mean())

    best = (mean_margin(suf), list(suf))
    start = best[0]
    for step in range(steps):
        # aggregate gradient over a random mini-batch of the training injections
        mb = rng.sample(range(len(tids)), min(16, len(tids)))
        grad = torch.zeros(n_suffix, len(vocab))
        for i in mb:
            pre, s, tail = det.build(tids[i], suf)
            oh = torch.zeros(len(s), E_table.size(0), requires_grad=True)
            with torch.no_grad():
                oh.data[range(len(s)), s] = 1.0
            e = torch.cat([E_table[torch.tensor(pre)], oh @ E_table, E_table[torch.tensor(tail)]], 0)[None]
            attn = torch.ones(1, e.size(1), dtype=torch.long)
            mk = torch.tensor([det.markers]) if hasattr(det, "markers") else None
            det.margin_from_embeds(e, attn, mk)[0].backward()
            grad += oh.grad[:, vocab]
        cand_tok = (-grad).topk(topk, dim=1).indices
        trials = []
        for _ in range(batch):
            pos = rng.randrange(n_suffix)
            c = suf.copy(); c[pos] = int(vocab[cand_tok[pos, rng.randrange(topk)]])
            trials.append(c)
        # score each trial by mean margin over the SAME mini-batch
        scores = []
        for c in trials:
            scores.append(float(_batch_margin(det, [det.build(tids[i], c) for i in mb]).mean()))
        j = int(torch.tensor(scores).argmin())
        suf = trials[j]
        full = mean_margin(suf)
        if full < best[0] and roundtrip_ok(det.tok, suf):
            best = (full, list(suf))
        if step % 20 == 0:
            log(f"  [univ] step {step:3d} mean-margin {best[0]:+.3f} (start {start:+.3f})")
    return {"suffix": suffix_str(det.tok, best[1]), "train_mean_margin": best[0], "start": start}
