#!/usr/bin/env python
"""Probe: SmilesRnn prototype — timing, sampling validity, determinism.

Validates the design before writing src/pharmaml/generative/rnn.py:
char-level GRU, CPU-only, teacher forcing, torch.Generator-based sampling.
"""

from __future__ import annotations

import time

import torch
from torch import nn
from torch.nn import functional as F

CORPUS_PATH = "scratch/probe_corpus_list.py"


def load_corpus() -> list[str]:
    ns = {}
    with open(CORPUS_PATH, encoding="utf-8") as fh:
        exec(fh.read(), ns)  # noqa: S102 - probe-only
    return ns["DEFAULT_CORPUS"]


PAD, UNK, START, END = "<pad>", "<unk>", "<start>", "<end>"


class Rnn(nn.Module):
    def __init__(self, vocab: list[str], emb=48, hidden=96, layers=1):
        super().__init__()
        self.vocab = vocab
        self.emb = nn.Embedding(len(vocab), emb, padding_idx=0)
        self.gru = nn.GRU(emb, hidden, layers, batch_first=True)
        self.head = nn.Linear(hidden, len(vocab))

    def forward(self, x, state=None):
        out, state = self.gru(self.emb(x), state)
        return self.head(out), state


def build_vocab(smiles_list: list[str]) -> list[str]:
    chars = sorted({c for s in smiles_list for c in s})
    return [PAD, UNK, START, END, *chars]


def encode(smiles: str, vocab: dict[str, int]) -> list[int]:
    return [vocab[START]] + [vocab.get(c, vocab[UNK]) for c in smiles] + [vocab[END]]


def train(smiles_list, epochs=8, batch_size=32, hidden=96, seed=42, lr=0.005):
    torch.manual_seed(seed)
    vocab = build_vocab(smiles_list)
    vmap = {t: i for i, t in enumerate(vocab)}
    model = Rnn(vocab, hidden=hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    seqs = [encode(s, vmap) for s in smiles_list]
    order = torch.randperm(len(seqs), generator=torch.Generator().manual_seed(seed))
    losses = []
    for epoch in range(epochs):
        perm = order if epoch == 0 else torch.randperm(
            len(seqs), generator=torch.Generator().manual_seed(seed + epoch)
        )
        tot, cnt = 0.0, 0
        for i in range(0, len(seqs), batch_size):
            batch = [seqs[j] for j in perm[i : i + batch_size].tolist()]
            maxlen = max(len(s) for s in batch)
            x = torch.zeros(len(batch), maxlen - 1, dtype=torch.long)
            y = torch.zeros(len(batch), maxlen - 1, dtype=torch.long)
            for r, s in enumerate(batch):
                x[r, : len(s) - 1] = torch.tensor(s[:-1])
                y[r, : len(s) - 1] = torch.tensor(s[1:])
            logits, _ = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, len(vocab)), y.reshape(-1), ignore_index=0
            )
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(loss) * len(batch)
            cnt += len(batch)
        losses.append(tot / cnt)
    return model, vmap, losses


@torch.no_grad()
def sample(model, vmap, n, temperature=1.0, seed=0, max_length=100):
    ivocab = {i: t for t, i in vmap.items()}
    banned = {vmap[PAD], vmap[UNK], vmap[START]}
    gen = torch.Generator().manual_seed(seed)
    state = None
    out = []
    cur = torch.full((n, 1), vmap[START], dtype=torch.long)
    done = [False] * n
    buf = [[] for _ in range(n)]
    for _ in range(max_length):
        logits, state = model(cur, state)
        logits = logits[:, -1, :].float() / max(temperature, 1e-6)
        probs = F.softmax(logits, dim=-1)
        probs[..., list(banned)] = 0.0
        probs = probs / probs.sum(-1, keepdim=True)
        nxt = torch.multinomial(probs, 1, generator=gen)
        for r in range(n):
            if done[r]:
                continue
            tok = int(nxt[r, 0])
            if tok == vmap[END]:
                done[r] = True
                continue
            buf[r].append(ivocab[tok])
        cur = nxt
        if all(done):
            break
    return ["".join(b) for b in buf]


def valid_fraction(smiles_list) -> float:
    from rdkit import Chem

    ok = [s for s in smiles_list if Chem.MolFromSmiles(s) is not None]
    return len(ok) / max(len(smiles_list), 1)


def main() -> int:
    corpus = load_corpus()
    print(f"corpus: {len(corpus)}")

    t0 = time.perf_counter()
    model, vmap, losses = train(corpus, epochs=8, hidden=96, seed=42)
    t1 = time.perf_counter()
    print(f"train 8 epochs hidden=96: {t1 - t0:.2f}s losses={['%.3f' % l for l in losses]}")

    t2 = time.perf_counter()
    smis = sample(model, vmap, 64, temperature=1.0, seed=1)
    t3 = time.perf_counter()
    vf = valid_fraction(smis)
    print(f"sample 64: {t3 - t2:.2f}s valid={vf:.2f}")
    print("examples:", smis[:8])

    # determinism: same training seed -> identical weights?
    model2, _, _ = train(corpus, epochs=8, hidden=96, seed=42)
    same = all(
        torch.equal(pa, pb)
        for pa, pb in zip(model.parameters(), model2.parameters())
    )
    print(f"train determinism (weights): {same}")

    s1 = sample(model, vmap, 32, seed=7)
    s2 = sample(model, vmap, 32, seed=7)
    print(f"sample determinism: {s1 == s2}")

    # smaller hidden / subset timing for tests
    t4 = time.perf_counter()
    m3, _, l3 = train(corpus[:60], epochs=2, hidden=64, seed=0)
    t5 = time.perf_counter()
    smis3 = sample(m3, vmap, 32, temperature=1.0, seed=3)
    print(f"tiny train (60 smiles, 2 epochs, h=64): {t5 - t4:.2f}s "
          f"losses={['%.3f' % l for l in l3]} valid={valid_fraction(smis3):.2f}")
    print("tiny examples:", smis3[:6])

    # temperature sweep on the good model
    for temp in (0.5, 0.8, 1.0, 1.2, 1.5):
        s = sample(model, vmap, 64, temperature=temp, seed=5)
        uniq = len(set(s))
        print(f"temp={temp}: valid={valid_fraction(s):.2f} unique={uniq}/64")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
