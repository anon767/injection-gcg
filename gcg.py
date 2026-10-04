"""Greedy Coordinate Gradient (Zou et al. 2023, arXiv:2307.15043) against a prompt-injection detector.

This is a white-box adversarial-robustness probe: given full access to a detector's weights (the
setting for any open-weights model on Hugging Face), it searches for a short suffix appended to an
injection so the detector scores it as benign. The point of the exercise is defensive — to show that a
learned text classifier placed in front of an LLM is not a security boundary, because the attacker who
has the weights optimises directly against it.

Any detector in detectors.py works: GCG only needs the embedding table and a differentiable forward
pass to the injection-vs-benign margin. Success is re-checked on the decoded string (re-tokenised from
scratch), so a reported bypass is exactly the text an attacker would send.
"""
import random
import time

import torch


def allowed_tokens(tok):
    """Word-initial, printable-ASCII, non-special tokens. Keeping suffix tokens word-initial makes
    decode->re-encode stable, so the optimised ids are the ids the detector actually scores."""
    special = set(tok.all_special_ids)
    ok = []
    for i in range(len(tok)):
        if i in special:
            continue
        p = tok.convert_ids_to_tokens(i)
        if not isinstance(p, str) or p[:1] not in ("Ġ", "▁") or len(p) < 2:
            continue
        s = p[1:]
        if s.isascii() and s.isprintable() and not any(c.isspace() for c in s):
            ok.append(i)
    return torch.tensor(ok)


def suffix_str(tok, ids):
    return tok.decode(ids, clean_up_tokenization_spaces=False)


def roundtrip_ok(tok, ids):
    return tok(suffix_str(tok, ids), add_special_tokens=False)["input_ids"] == list(ids)


def _onehot_embed(det, ids_list, suffix_onehot, E_table):
    """Build a batch of sequences where the suffix slot is a (possibly soft) one-hot @ embedding table,
    so gradients flow into suffix_onehot. ids_list items: (pre, suf, tail)."""
    dev = E_table.device
    L = 0
    for pre, suf, tail in ids_list:
        L = max(L, len(pre) + len(suf) + len(tail))
    attn = torch.zeros(len(ids_list), L, dtype=torch.long, device=dev)
    out = torch.zeros(len(ids_list), L, E_table.size(1), device=dev)
    for b, (pre, suf, tail) in enumerate(ids_list):
        n = len(pre) + len(suf) + len(tail)
        pe = E_table[torch.tensor(pre, device=dev)]
        se = suffix_onehot @ E_table
        te = E_table[torch.tensor(tail, device=dev)]
        out[b, :n] = torch.cat([pe, se, te], 0)
        attn[b, :n] = 1
    return out, attn


def attack(det, text, n_suffix=20, steps=250, topk=256, batch=256, tau=0.0, seed=0, log=print,
           early_stop=True):
    """Return dict with the best suffix string and whether it bypasses the detector (margin<tau).

    tau is the detector's operating threshold in margin space (logit injection - benign); success
    means driving the margin below it. `trajectory` is the best margin reached by each step, so a
    single full-length run (early_stop=False) yields the whole attack-success-vs-budget curve.
    """
    rng = random.Random(seed)
    torch.manual_seed(seed)
    E_table = det.embed_matrix().detach()
    dev = E_table.device
    vocab = allowed_tokens(det.tok).to(dev)

    tid = det.tok(text.replace(getattr(det.tok, "mask_token", "\0"), " "),
                  add_special_tokens=False)["input_ids"]
    suf = vocab[torch.randint(0, len(vocab), (n_suffix,), generator=torch.Generator().manual_seed(seed)).to(dev)].tolist()

    def split(suffix):
        return det.build(tid, suffix)

    @torch.no_grad()
    def margin_of(suffix):
        pre, s, tail = split(suffix)
        oh = torch.zeros(1, len(s), E_table.size(0), device=dev); oh[0, range(len(s)), s] = 1.0
        E, attn = _onehot_embed(det, [(pre, s, tail)], oh[0], E_table)
        return float(det.margin_from_embeds(E, attn, _markers(det, 1)))

    def _markers(det, b):
        return torch.tensor([det.markers] * b) if hasattr(det, "markers") else None

    best = (margin_of(suf), list(suf))
    start = best[0]
    trajectory = []
    for step in range(steps):
        pre, s, tail = split(suf)
        oh = torch.zeros(len(s), E_table.size(0), device=dev, requires_grad=True)
        with torch.no_grad():
            oh.data[range(len(s)), s] = 1.0
        E, attn = _onehot_embed(det, [(pre, s, tail)], oh, E_table)
        _markers(det, 1)
        m = det.margin_from_embeds(E, attn, _markers(det, 1))[0]
        m.backward()
        grad = oh.grad[:, vocab]                       # [n_suffix, |vocab|]: d margin / d token
        cand_tok = (-grad).topk(topk, dim=1).indices   # tokens that most *decrease* the margin

        # sample `batch` single-token swaps, evaluate exactly, keep the best
        trials = []
        for _ in range(batch):
            pos = rng.randrange(len(s))
            newtok = int(vocab[cand_tok[pos, rng.randrange(topk)]])
            cand = s.copy(); cand[pos] = newtok
            trials.append(cand)
        # batch-evaluate trials
        rows = [split(c) for c in trials]
        ohs = []
        for (p, sc, t), c in zip(rows, trials):
            o = torch.zeros(len(sc), E_table.size(0), device=dev); o[range(len(sc)), sc] = 1.0
            ohs.append((p, sc, t, o))
        with torch.no_grad():
            L = max(len(p) + len(sc) + len(t) for p, sc, t, _ in ohs)
            Eb = torch.zeros(len(ohs), L, E_table.size(1), device=dev); ab = torch.zeros(len(ohs), L, dtype=torch.long, device=dev)
            for b, (p, sc, t, o) in enumerate(ohs):
                e = torch.cat([E_table[torch.tensor(p, device=dev)], o @ E_table, E_table[torch.tensor(t, device=dev)]], 0)
                Eb[b, :e.size(0)] = e; ab[b, :e.size(0)] = 1
            ms = det.margin_from_embeds(Eb, ab, _markers(det, len(ohs)))
        j = int(ms.argmin())
        if float(ms[j]) < best[0]:
            best = (float(ms[j]), trials[j])
        # move current suffix toward the best trial of this step (standard GCG greedy update)
        suf = trials[j] if float(ms[j]) <= margin_of(suf) else suf
        trajectory.append(best[0])
        if step % 25 == 0:
            log(f"  step {step:3d} margin {best[0]:+.3f} (start {start:+.3f})  suffix={suffix_str(det.tok, best[1])!r}")
        if early_stop and best[0] < tau and roundtrip_ok(det.tok, best[1]):
            break

    bs = suffix_str(det.tok, best[1])
    # final check on the STRING (what an attacker actually sends)
    str_margin = float(det.margin([text + bs])[0])
    return {"text": text, "suffix": bs, "n_suffix": n_suffix, "steps_run": step + 1, "tau": tau,
            "start_margin": start, "suffix_ids_margin": best[0], "string_margin": str_margin,
            "bypass": str_margin < tau, "roundtrip": roundtrip_ok(det.tok, best[1]),
            "trajectory": trajectory}
