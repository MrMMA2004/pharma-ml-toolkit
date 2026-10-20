#!/usr/bin/env python
"""Probe: RL fine-tuning loop prototype (REINFORCE + baseline + replay + prior anchor).

Verifies the algorithm actually improves the objective before writing rl.py,
and measures per-iteration wall time.
"""

from __future__ import annotations

import importlib.util
import time

import torch
from torch import nn
from torch.nn import functional as F
from rdkit import Chem
from rdkit.Chem.QED import qed

_spec = importlib.util.spec_from_file_location(
    "probe_rnn", "scratch/probe_generative_rnn.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
END, PAD, START = _mod.END, _mod.PAD, _mod.START
Rnn, build_vocab, encode = _mod.Rnn, _mod.build_vocab, _mod.encode
load_corpus, sample, train = _mod.load_corpus, _mod.sample, _mod.train


def canonicalize(smiles_list):
    out = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            out.append(None)
        else:
            out.append(Chem.MolToSmiles(mol))
    return out


def score(smiles_list):
    out = []
    for s in smiles_list:
        if s is None:
            out.append(0.0)
            continue
        mol = Chem.MolFromSmiles(s)
        out.append(qed(mol) if mol is not None else 0.0)
    return out


def sequence_logprob(model, vmap, seqs) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-sequence logprob + mean per-step entropy (differentiable)."""
    enc = [encode(s, vmap) for s in seqs]
    maxlen = max(len(e) for e in enc)
    x = torch.zeros(len(enc), maxlen - 1, dtype=torch.long)
    y = torch.zeros(len(enc), maxlen - 1, dtype=torch.long)
    mask = torch.zeros(len(enc), maxlen - 1)
    for r, e in enumerate(enc):
        x[r, : len(e) - 1] = torch.tensor(e[:-1])
        y[r, : len(e) - 1] = torch.tensor(e[1:])
        mask[r, : len(e) - 1] = 1.0
    logits, _ = model(x)
    logp = F.log_softmax(logits, dim=-1)
    tok_lp = logp.gather(-1, y.unsqueeze(-1)).squeeze(-1)
    seq_lp = (tok_lp * mask).sum(-1)
    ent = -(logp.exp() * logp).sum(-1)
    mean_ent = (ent * mask).sum() / mask.sum()
    return seq_lp, mean_ent


def main() -> int:
    corpus = load_corpus()
    torch.manual_seed(0)
    prior, vmap, _ = train(corpus, epochs=10, hidden=96, seed=123)
    agent = Rnn(prior.vocab, hidden=96)
    agent.load_state_dict(prior.state_dict())
    opt = torch.optim.Adam(agent.parameters(), lr=0.003)

    sigma, beta, batch, iters = 0.0, 0.02, 32, 20
    anchor = 0.1
    baseline = None
    replay: list[tuple[str, float]] = []

    t_start = time.perf_counter()
    for it in range(iters):
        raw = sample(agent, vmap, batch, temperature=0.9, seed=1000 + it)
        canon = canonicalize(raw)
        scores = score(canon)
        valid = [c is not None for c in canon]

        # replay bookkeeping (top scoring valid unique)
        for c, s in zip(canon, scores):
            if c is not None and s > 0:
                replay.append((c, s))
        replay = sorted(replay, key=lambda t: -t[1])
        seen = set()
        uniq = []
        for c, s in replay:
            if c not in seen:
                seen.add(c)
                uniq.append((c, s))
        replay = uniq[:32]

        mean_score = sum(scores) / len(scores)
        baseline = mean_score if baseline is None else 0.9 * baseline + 0.1 * mean_score

        # training batch: ALL sampled strings (invalid ones carry score 0 and
        # get pushed DOWN) + replay mix
        train_seqs = list(zip(raw, scores))
        train_seqs += replay[: batch // 2]
        seqs = [t[0] for t in train_seqs]
        scs = torch.tensor([t[1] for t in train_seqs])

        with torch.no_grad():
            prior_lp, _ = sequence_logprob(prior, vmap, seqs)
        agent_lp, entropy = sequence_logprob(agent, vmap, seqs)
        augmented = (1 - sigma) * agent_lp + sigma * prior_lp
        adv = scs - baseline
        adv = (adv - adv.mean()) / (adv.std() + 1e-6)  # normalized advantage
        anchor_loss = ((agent_lp - prior_lp) ** 2).mean()
        loss = -(adv * augmented).mean() - beta * entropy + anchor * anchor_loss
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
        opt.step()

        best = max(scores)
        print(
            f"it={it:2d} valid={sum(valid)}/{batch} mean={mean_score:.3f} "
            f"best={best:.3f} replay={len(replay)} loss={float(loss):+.3f} "
            f"anchor={float(anchor_loss):.3f}"
        )

    t_end = time.perf_counter()
    print(f"total RL time: {t_end - t_start:.2f}s "
          f"({(t_end - t_start) / iters:.2f}s/iter)")

    # also invalid sequences should be pushed DOWN: track invalid lp over time?
    final_raw = sample(agent, vmap, 64, temperature=0.9, seed=999)
    final_canon = canonicalize(final_raw)
    print(f"final valid fraction (64 samples): "
          f"{sum(c is not None for c in final_canon) / 64:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
