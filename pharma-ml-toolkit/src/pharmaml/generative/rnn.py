"""Character-level SMILES RNN -- the "prior" of the generative toolkit.

A small GRU language model over single SMILES characters (embedding 48,
hidden 96, one layer by default), deliberately CPU-sized: the full
:mod:`pharmaml.generative` story -- train a prior on the bundled corpus,
fine-tune it with reinforcement learning -- runs in seconds on a laptop
CPU.

Vocabulary: one token per character seen during training plus four
special tokens (``<pad>``, ``<unk>``, ``<start>``, ``<end>``).  Sampling
starts from ``<start>``, draws characters from the softmax distribution
(temperature-controlled) and stops at ``<end>`` or the length cap;
``<pad>`` / ``<unk>`` / ``<start>`` are never emitted.

PyTorch is an *optional* dependency imported lazily, mirroring
:mod:`pharmaml.ml.neural`: importing this module (or ``pharmaml`` itself)
never requires torch; the import happens at :meth:`SmilesRnn.train_prior`
/ :meth:`SmilesRnn.sample` time and raises an actionable
:class:`~pharmaml.exceptions.DependencyError` when missing::

    pip install 'pharmaml[deep]'

Determinism: ``train_prior`` seeds ``torch.manual_seed`` from the
constructor ``seed`` and uses dedicated ``torch.Generator`` objects for
batch shuffling; ``sample`` accepts an explicit ``seed`` (a fresh local
generator, independent of any previous call).  Two identically-seeded
runs on the same CPU produce identical weights and identical samples.

Example
-------
>>> from pharmaml.generative import SmilesRnn, DEFAULT_CORPUS
>>> prior = SmilesRnn(hidden_dim=96, seed=42)
>>> losses = prior.train_prior(DEFAULT_CORPUS, epochs=8)
>>> smiles = prior.sample(32, temperature=0.9, seed=1)
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from pharmaml.exceptions import DataValidationError, DependencyError
from pharmaml.logger import get_logger

__all__ = ["SmilesRnn"]

_LOG = get_logger(__name__)

#: pip command embedded in the DependencyError raised when torch is missing.
_TORCH_HINT = (
    "generative RNN models require torch — "
    "pip install 'pharmaml[deep]' (or pip install torch)"
)

#: Special vocabulary tokens (never emitted by the sampler).
PAD, UNK, START, END = "<pad>", "<unk>", "<start>", "<end>"
_SPECIAL_TOKENS = (PAD, UNK, START, END)

#: Seeds accepted by ``torch.manual_seed`` / ``torch.Generator.manual_seed``
#: (the closed interval [-2**63, 2**64 - 1]); anything outside raises a raw
#: torch ``ValueError: Overflow when unpacking long long``, so the range is
#: enforced with a typed error instead.
_TORCH_SEED_MIN, _TORCH_SEED_MAX = -(2**63), 2**64 - 1


def _validate_torch_seed(seed: int, what: str) -> int:
    """Raise a typed error unless ``seed`` is a torch-encodable integer."""
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise DataValidationError(
            f"{what} must be an int, got {seed!r}"
        )
    if not _TORCH_SEED_MIN <= seed <= _TORCH_SEED_MAX:
        raise DataValidationError(
            f"{what}={seed} is outside the range torch can seed "
            f"[{_TORCH_SEED_MIN}, {_TORCH_SEED_MAX}] (2**64 - 1)"
        )
    return seed


def _require_torch() -> Any:
    """Import and return torch, or raise the actionable error."""
    try:
        import torch
    except ImportError as exc:
        raise DependencyError(_TORCH_HINT) from exc
    return torch


class SmilesRnn:
    """Character-level GRU SMILES generator (CPU-only, torch optional).

    Parameters
    ----------
    embedding_dim:
        Character embedding size (default 48).
    hidden_dim:
        GRU hidden state size (default 96).
    n_layers:
        GRU layer count (default 1).
    seed:
        Seed for weight initialisation and batch shuffling (default 42).
    device:
        Always CPU -- the parameter exists only to make the contract
        explicit; anything other than ``"cpu"`` raises.

    Raises
    ------
    DataValidationError
        At construction time for out-of-range dimensions / ``device`` or a
        ``seed`` outside ``[-2**63, 2**64 - 1]`` (the interval torch's RNG
        can encode); at *method* call time (never at import/construction)
        when torch is not installed.
    """

    def __init__(
        self,
        *,
        embedding_dim: int = 48,
        hidden_dim: int = 96,
        n_layers: int = 1,
        seed: int = 42,
        device: str = "cpu",
    ) -> None:
        if device != "cpu":
            raise DataValidationError(
                f"SmilesRnn is CPU-only, got device={device!r}"
            )
        _validate_torch_seed(seed, "seed")
        if embedding_dim < 4 or hidden_dim < 8:
            raise DataValidationError(
                f"embedding_dim/hidden_dim too small ({embedding_dim}/{hidden_dim}) "
                "-- use at least 4/8"
            )
        if not 1 <= n_layers <= 4:
            raise DataValidationError(f"n_layers must be in [1, 4], got {n_layers}")
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.seed = seed
        self.device = "cpu"
        self._vocab: list[str] | None = None
        self._model: Any = None  # torch.nn.Module, built by train_prior

    # -- vocabulary ------------------------------------------------------
    @property
    def vocab(self) -> list[str] | None:
        """Token vocabulary (``None`` until :meth:`train_prior` runs)."""
        return self._vocab

    @property
    def is_trained(self) -> bool:
        """True once a prior has been trained (model weights exist)."""
        return self._model is not None

    def _build_vocab(self, smiles_list: Sequence[str]) -> list[str]:
        """Special tokens + the sorted character alphabet of the corpus."""
        chars = sorted({char for smiles in smiles_list for char in str(smiles)})
        vocab = [*_SPECIAL_TOKENS, *chars]
        seen: set[str] = set()
        unique = []
        for token in vocab:
            if token not in seen:
                seen.add(token)
                unique.append(token)
        return unique

    def _token_map(self) -> dict[str, int]:
        """Token -> index map for the current vocabulary."""
        if self._vocab is None:
            raise DataValidationError("no vocabulary yet -- call train_prior first")
        return {token: index for index, token in enumerate(self._vocab)}

    def _encode(self, smiles: str) -> list[int]:
        """Token indices of ``smiles`` (unknown characters -> ``<unk>``)."""
        mapping = self._token_map()
        tokens = [START] + [char for char in str(smiles)] + [END]
        return [mapping.get(token, mapping[UNK]) for token in tokens]

    # -- model construction ----------------------------------------------
    def _build_model(self, torch: Any) -> Any:
        """Construct the (untrained) GRU network for the current vocab."""
        vocab = self._vocab
        if vocab is None:
            raise DataValidationError("no vocabulary yet -- call train_prior first")

        class _Net(torch.nn.Module):
            def __init__(self, n_vocab: int, emb: int, hidden: int, layers: int):
                super().__init__()
                self.emb = torch.nn.Embedding(n_vocab, emb, padding_idx=0)
                self.gru = torch.nn.GRU(emb, hidden, layers, batch_first=True)
                self.head = torch.nn.Linear(hidden, n_vocab)

            def forward(self, x, state=None):
                out, state = self.gru(self.emb(x), state)
                return self.head(out), state

        return _Net(len(vocab), self.embedding_dim, self.hidden_dim, self.n_layers)

    # -- public API ------------------------------------------------------
    def train_prior(
        self,
        smiles_list: Sequence[str],
        *,
        epochs: int = 8,
        batch_size: int = 32,
        lr: float = 0.005,
        grad_clip: float = 5.0,
    ) -> list[float]:
        """Teacher-forcing training of the prior; returns per-epoch losses.

        The vocabulary is built on the first call; later calls reuse it so
        the same object can be fine-tuned incrementally (characters absent
        from the original vocabulary encode as ``<unk>``).

        Raises
        ------
        DataValidationError
            If ``smiles_list`` is empty, ``epochs``/``batch_size`` are
            out of range, or ``lr``/``grad_clip`` are not positive finite
            numbers (``lr=0`` silently trains nothing and ``lr=inf`` NaNs
            every weight -- both are configuration mistakes worth naming).
        DependencyError
            If torch is not installed.
        """
        torch = _require_torch()
        from torch.nn import functional as F

        data = [str(smiles) for smiles in smiles_list if str(smiles).strip()]
        if not data:
            raise DataValidationError("smiles_list is empty -- nothing to train on")
        if epochs < 1:
            raise DataValidationError(f"epochs must be >= 1, got {epochs}")
        if batch_size < 1:
            raise DataValidationError(f"batch_size must be >= 1, got {batch_size}")
        for name, value in (("lr", lr), ("grad_clip", grad_clip)):
            if (
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise DataValidationError(
                    f"{name} must be a positive finite number, got {value!r}"
                )

        if self._vocab is None:
            self._vocab = self._build_vocab(data)
            _LOG.info(
                "prior vocabulary: %d tokens (%d corpus characters)",
                len(self._vocab),
                len(self._vocab) - len(_SPECIAL_TOKENS),
            )
        if self._model is None:
            torch.manual_seed(self.seed)
            self._model = self._build_model(torch)

        sequences = [self._encode(smiles) for smiles in data]
        optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)
        shuffle_gen = torch.Generator()
        shuffle_gen.manual_seed(self.seed + 1)

        losses: list[float] = []
        for epoch in range(epochs):
            order = torch.randperm(len(sequences), generator=shuffle_gen).tolist()
            total, count = 0.0, 0
            for start in range(0, len(order), batch_size):
                batch = [sequences[i] for i in order[start : start + batch_size]]
                width = max(len(seq) for seq in batch)
                x = torch.zeros(len(batch), width - 1, dtype=torch.long)
                y = torch.zeros(len(batch), width - 1, dtype=torch.long)
                for row, seq in enumerate(batch):
                    x[row, : len(seq) - 1] = torch.tensor(seq[:-1])
                    y[row, : len(seq) - 1] = torch.tensor(seq[1:])
                logits, _ = self._model(x)
                loss = F.cross_entropy(
                    logits.reshape(-1, len(self._vocab)),
                    y.reshape(-1),
                    ignore_index=0,  # <pad>
                )
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), grad_clip)
                optimizer.step()
                total += float(loss.detach()) * len(batch)
                count += len(batch)
            losses.append(total / count)
            _LOG.debug("prior epoch %d/%d loss %.4f", epoch + 1, epochs, losses[-1])
        return losses

    def copy(self) -> SmilesRnn:
        """Return a deep copy with fresh optimizer state (frozen prior etc.).

        The copy shares nothing mutable with this instance, so fine-tuning
        one never moves the other -- the RL agent / prior pattern.
        """
        torch = _require_torch()
        clone = SmilesRnn(
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
            seed=self.seed,
        )
        clone._vocab = list(self._vocab) if self._vocab is not None else None
        if self._model is not None:
            torch.manual_seed(self.seed)
            clone._model = clone._build_model(torch)
            clone._model.load_state_dict(self._model.state_dict())
            clone._model.eval()
        return clone

    def sample(
        self,
        n: int,
        *,
        temperature: float = 1.0,
        seed: int | None = None,
        max_length: int = 120,
    ) -> list[str]:
        """Draw ``n`` SMILES strings from the trained model.

        Parameters
        ----------
        n:
            Number of strings to generate.
        temperature:
            Softmax temperature (``> 0``; lower = more conservative).
        seed:
            Optional seed -- a *fresh local* generator, so repeated calls
            with the same seed are reproducible regardless of what was
            sampled before.  ``None`` advances a persistent generator
            seeded at construction.
        max_length:
            Hard character cap per molecule.

        Returns
        -------
        list[str]
            Raw generated strings -- invalid SMILES are possible (and
            expected); scoring objectives map them to 0.0.

        Raises
        ------
        DataValidationError
            If the model is untrained, ``n``/``max_length`` are out of
            range, ``temperature`` is not a positive finite number (NaN
            passes ``<= 0`` comparisons and crashes torch's multinomial;
            +inf degrades the softmax to noise), or ``seed`` is outside
            ``[-2**63, 2**64 - 1]`` (the torch-encodable interval).
        DependencyError
            If torch is not installed.
        """
        if self._model is None:
            raise DataValidationError(
                "the prior is untrained -- call train_prior() before sample()"
            )
        if n < 1:
            raise DataValidationError(f"n must be >= 1, got {n}")
        if not 0 < max_length <= 2000:
            raise DataValidationError(f"max_length must be in (0, 2000], got {max_length}")
        if (
            not isinstance(temperature, (int, float))
            or not math.isfinite(temperature)
            or temperature <= 0
        ):
            raise DataValidationError(
                f"temperature must be a positive finite number, got {temperature!r}"
            )
        if seed is not None:
            _validate_torch_seed(seed, "seed")
        torch = _require_torch()
        from torch.nn import functional as F

        mapping = self._token_map()
        vocab = self._vocab
        if vocab is None:  # unreachable after _token_map(); narrows the type
            raise DataValidationError("no vocabulary yet -- call train_prior first")
        banned = [mapping[t] for t in (PAD, UNK, START)]

        if seed is not None:
            generator = torch.Generator()
            generator.manual_seed(int(seed))
        else:
            if not hasattr(self, "_sample_gen"):
                self._sample_gen = torch.Generator()
                self._sample_gen.manual_seed(self.seed + 2)
            generator = self._sample_gen

        with torch.no_grad():
            state = None
            current = torch.full((n, 1), mapping[START], dtype=torch.long)
            finished = [False] * n
            buffers: list[list[str]] = [[] for _ in range(n)]
            for _ in range(max_length):
                logits, state = self._model(current, state)
                step = logits[:, -1, :].float() / float(temperature)
                probs = F.softmax(step, dim=-1)
                probs[:, banned] = 0.0
                probs = probs / probs.sum(dim=-1, keepdim=True)
                nxt = torch.multinomial(probs, 1, generator=generator)
                for row in range(n):
                    if finished[row]:
                        continue
                    token = vocab[int(nxt[row, 0])]
                    if token == END:
                        finished[row] = True
                        continue
                    buffers[row].append(token)
                current = nxt
                if all(finished):
                    break
        return ["".join(tokens) for tokens in buffers]

    # -- package-private helpers (used by rl.py) --------------------------
    def _padded_batch(
        self, torch: Any, sequences: list[list[int]]
    ) -> tuple[Any, Any, Any]:
        """Teacher-forcing tensors ``(x, y, mask)`` for encoded sequences."""
        width = max(len(seq) for seq in sequences)
        x = torch.zeros(len(sequences), width - 1, dtype=torch.long)
        y = torch.zeros(len(sequences), width - 1, dtype=torch.long)
        mask = torch.zeros(len(sequences), width - 1)
        for row, seq in enumerate(sequences):
            x[row, : len(seq) - 1] = torch.tensor(seq[:-1])
            y[row, : len(seq) - 1] = torch.tensor(seq[1:])
            mask[row, : len(seq) - 1] = 1.0
        return x, y, mask

    def _sequence_logprob_and_entropy(
        self, torch: Any, sequences: list[list[int]]
    ) -> tuple[Any, Any]:
        """Per-sequence log-probability + mean per-step distribution entropy.

        Both tensors are differentiable through the model -- the RL loss
        is built from them.
        """
        from torch.nn import functional as F

        x, y, mask = self._padded_batch(torch, sequences)
        logits, _ = self._model(x)
        log_probs = F.log_softmax(logits, dim=-1)
        token_lp = log_probs.gather(-1, y.unsqueeze(-1)).squeeze(-1)
        seq_lp = (token_lp * mask).sum(dim=-1)
        entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
        mean_entropy = (entropy * mask).sum() / mask.sum()
        return seq_lp, mean_entropy

    def _encoded(self, smiles: str) -> list[int]:
        """Encoded token indices (package-private alias of ``_encode``)."""
        return self._encode(smiles)

    def _vocab_size(self) -> int:
        """Vocabulary size (package-private)."""
        if self._vocab is None:
            raise DataValidationError("no vocabulary yet -- call train_prior first")
        return len(self._vocab)

    def _apply_sgd_step(self, torch: Any, optimizer: Any, loss: Any, grad_clip: float) -> None:
        """Zero-grad -> backward -> clip -> step (package-private)."""
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), grad_clip)
        optimizer.step()
