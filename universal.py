"""Universal GCG suffix: optimise ONE suffix over many injections at once (Zou et al. 2023, sec 3.2),
then measure its bypass rate on held-out injections. This is the stronger claim: a single fixed string
an attacker can paste onto any injection, not a per-prompt re-optimisation.

Candidate scoring is string-faithful (decode -> retokenise -> det.margin), exactly like gcg.attack:
id-space margins under-report/over-report vs the deployed detector because decode->re-encode is not
id-stable. Gradients for candidate proposals stay in id space (first-order heuristic, fine)."""
import random

import torch

from gcg import allowed_tokens, suffix_str, _markers, _onehot_embed


def universal(det, texts, n_suffix=20, steps=200, topk=256, batch=192, seed=0, log=print):
    rng = random.Random(seed)
    E_table = det.embed_matrix().detach()
    dev = E_table.device
    vocab = allowed_tokens(det.tok).to(dev)
    tids = [det.tok(t.replace(getattr(det.tok, "mask_token", "\0"), " "),
                    add_special_tokens=False)["input_ids"] for t in texts]
    suf = vocab[torch.randint(0, len(vocab), (n_suffix,),
                              generator=torch.Generator().manual_seed(seed)).to(dev)].tolist()

    def to_str(suffix):
        return suffix_str(det.tok, suffix)

    def mean_string_margin(suffix, idx=None):
        """Mean of det.margin over texts[idx] (all, if idx=None) with the decoded suffix appended."""
        idx = range(len(texts)) if idx is None else idx
        return float(det.margin([texts[i] + to_str(suffix) for i in idx]).mean())

    best = (mean_string_margin(suf), list(suf))
    start = best[0]
    for step in range(steps):
        # aggregate id-space gradient over a random mini-batch of the training injections
        mb = rng.sample(range(len(tids)), min(16, len(tids)))
        rows = [det.build(tids[i], suf) for i in mb]
        oh = torch.zeros(n_suffix, E_table.size(0), device=dev, requires_grad=True)
        with torch.no_grad():
            oh[range(n_suffix), suf] = 1.0
        E, attn = _onehot_embed(det, rows, oh, E_table)
        det.margin_from_embeds(E, attn, _markers(det, len(rows))).mean().backward()
        cand_tok = (-oh.grad[:, vocab]).topk(min(topk, len(vocab)), dim=1).indices

        trials = []
        for _ in range(batch):
            pos = rng.randrange(n_suffix)
            c = list(suf)
            c[pos] = int(vocab[cand_tok[pos, rng.randrange(cand_tok.size(1))]])
            trials.append(c)
        # score each trial EXACTLY (string level) on the SAME mini-batch
        scores = [mean_string_margin(c, mb) for c in trials]
        j = min(range(len(trials)), key=lambda k: scores[k])
        suf = trials[j]
        full = mean_string_margin(suf)
        if full < best[0]:
            best = (full, list(suf))
        if step % 20 == 0:
            log(f"  [univ] step {step:3d} mean-margin {best[0]:+.3f} (start {start:+.3f})")
    return {"suffix": to_str(best[1]), "train_mean_margin": best[0], "start": start}
