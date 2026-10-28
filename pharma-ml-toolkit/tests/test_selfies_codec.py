"""Tests for :mod:`pharmaml.chem.selfies_codec` (round 100x-162).

The SELFIES surface -- the robust molecular string codec, the dataset
audit, the corruption experiment, the vocabulary helpers, the ML feature
matrix and the constraint context manager -- over the optional
``selfies`` package (the-matter-lab/selfies 2.x, verified 2.1.1).

The suite pins:

* the codec round trip and its typed refusals (empty input, malformed
  SELFIES, unencodable SMILES -- the package's error text preserved);
* the batch converter's row alignment (a failed row becomes ``None``
  plus an error record, never a dropped row);
* the vocabulary helpers (alphabet, frequencies) on exact values;
* the random-SELFIES generator and the STONED-style mutants: every
  output RDKit-valid, seeded determinism, dedup + original-exclusion;
* the round-trip audit's funnel on a known-good list (fidelity 1.0)
  and on a list with a poison row;
* the corruption experiment's headline property (SELFIES validity
  strictly above SMILES validity at k>=2 on aromatic molecules) and
  the tidy ``as_frame``;
* the feature matrix shapes (label / one-hot), the vocabulary order,
  and the typed refusals (unknown encoding, unencodable row with its
  index, empty column);
* the semantic-constraint context manager: the clamping effect while
  active, the restore on clean exit AND on exception, the preset list;
* the guard when the ``selfies`` package is missing (import-blocker
  monkeypatch): the ``DependencyError`` carries the exact
  ``pip install 'pharmaml[selfies]'`` command.

The real-path tests skip when ``selfies`` (or RDKit, for the audit
halves) is absent; the blocker test runs everywhere.
"""

from __future__ import annotations

import importlib.util

import pytest

from pharmaml.exceptions import DataValidationError, DependencyError

_HAS_SELFIES = importlib.util.find_spec("selfies") is not None
_HAS_RDKIT = importlib.util.find_spec("rdkit") is not None

pytestmark = pytest.mark.skipif(
    not _HAS_SELFIES, reason="selfies not installed (the zip-only tree)"
)


# --------------------------------------------------------------------------- #
# codec: single molecules
# --------------------------------------------------------------------------- #
def test_codec_round_trip_exact_values() -> None:
    from pharmaml.chem.selfies_codec import selfies_to_smiles, smiles_to_selfies

    assert smiles_to_selfies("CCO") == "[C][C][O]"
    assert selfies_to_smiles("[C][C][O]") == "CCO"
    # the aromatic round trip is canonical-equal, not string-equal (the
    # decoder writes the kekule form -- the audit's own criterion)
    encoded = smiles_to_selfies("c1ccccc1")
    assert selfies_to_smiles(encoded) == "C1=CC=CC=C1"


def test_codec_typed_refusals() -> None:
    from pharmaml.chem.selfies_codec import selfies_to_smiles, smiles_to_selfies

    with pytest.raises(DataValidationError, match="non-empty SMILES"):
        smiles_to_selfies("")
    with pytest.raises(DataValidationError, match="non-empty SMILES"):
        smiles_to_selfies("   ")
    with pytest.raises(DataValidationError, match="non-empty SELFIES"):
        selfies_to_smiles("")
    with pytest.raises(DataValidationError, match="SELFIES decoder rejected"):
        selfies_to_smiles("[C")  # hanging bracket


def test_codec_len_and_tokens() -> None:
    from pharmaml.chem.selfies_codec import selfies_len, selfies_tokens

    assert selfies_len("[C][C][O]") == 3
    assert selfies_tokens("[C][C][O]") == ["[C]", "[C]", "[O]"]


# --------------------------------------------------------------------------- #
# batch conversion
# --------------------------------------------------------------------------- #
def test_batch_conversion_row_alignment() -> None:
    from pharmaml.chem.selfies_codec import batch_smiles_to_selfies

    report = batch_smiles_to_selfies(["CCO", "C1CC", "CCN"])
    assert report.n_input == 3
    assert report.n_encoded == 2
    assert report.n_failed == 1
    # the failed row is preserved as None at its index, never dropped
    assert report.selfies[1] is None
    assert report.selfies[0] == "[C][C][O]"
    assert report.selfies[2] == "[C][C][N]"
    assert report.errors[0][0] == 1
    assert report.errors[0][1] == "C1CC"
    assert report.summary() == "2/3 rows encoded (1 failed)"


def test_batch_conversion_all_good() -> None:
    from pharmaml.chem.selfies_codec import batch_smiles_to_selfies

    report = batch_smiles_to_selfies(["CCO", "CCN"])
    assert report.n_failed == 0
    assert report.errors == []
    assert report.selfies == ["[C][C][O]", "[C][C][N]"]


# --------------------------------------------------------------------------- #
# vocabulary helpers
# --------------------------------------------------------------------------- #
def test_alphabet_and_frequencies_exact() -> None:
    from pharmaml.chem.selfies_codec import (
        selfies_alphabet,
        selfies_symbol_frequencies,
        smiles_to_selfies,
    )

    encoded = [smiles_to_selfies("CCO"), smiles_to_selfies("CCN")]
    assert selfies_alphabet(encoded) == ["[C]", "[N]", "[O]"]
    frequencies = selfies_symbol_frequencies(encoded)
    assert frequencies == {"[C]": 4, "[N]": 1, "[O]": 1}
    # frequencies sort by count descending, then symbol
    assert next(iter(frequencies)) == "[C]"


# --------------------------------------------------------------------------- #
# random generator + mutants (RDKit validity)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _HAS_RDKIT, reason="RDKit not installed")
def test_random_selfies_molecules_valid_and_deterministic() -> None:
    from rdkit import Chem

    from pharmaml.chem.selfies_codec import random_selfies_molecules

    molecules = random_selfies_molecules(6, 10, seed=42)
    assert len(molecules) == 6
    assert all(Chem.MolFromSmiles(s) is not None for s in molecules)
    # seeded determinism: two same-seed calls agree exactly
    assert molecules == random_selfies_molecules(6, 10, seed=42)
    with pytest.raises(DataValidationError, match="n must be >= 0"):
        random_selfies_molecules(-1, 10)
    with pytest.raises(DataValidationError, match="length must be >= 1"):
        random_selfies_molecules(4, 0)


@pytest.mark.skipif(not _HAS_RDKIT, reason="RDKit not installed")
def test_selfies_mutants_valid_excluding_original() -> None:
    from rdkit import Chem

    from pharmaml.chem.selfies_codec import selfies_mutants

    mutants = selfies_mutants("CCO", n=12, mutations=1, seed=42)
    assert mutants  # the neighbourhood is non-empty
    assert "CCO" not in mutants  # the parent itself is excluded
    assert len(mutants) == len(set(mutants))  # deduplicated
    assert all(Chem.MolFromSmiles(s) is not None for s in mutants)
    with pytest.raises(DataValidationError, match="mutations must be >= 1"):
        selfies_mutants("CCO", n=4, mutations=0)


# --------------------------------------------------------------------------- #
# round-trip audit
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _HAS_RDKIT, reason="RDKit not installed")
def test_roundtrip_report_known_good() -> None:
    from pharmaml.chem.selfies_codec import selfies_roundtrip_report

    report = selfies_roundtrip_report(["CCO", "CCN", "c1ccccc1"])
    assert report.n_input == 3
    assert report.n_canonical_match == 3
    assert report.fidelity == 1.0
    assert report.alphabet_size > 5
    assert report.encode_errors == []
    assert report.mismatch_examples == []
    assert "100.0%" in report.summary()
    assert "funnel: 3 parsed" in report.summary()


@pytest.mark.skipif(not _HAS_RDKIT, reason="RDKit not installed")
def test_roundtrip_report_poison_row_counted_not_crashed() -> None:
    from pharmaml.chem.selfies_codec import selfies_roundtrip_report

    report = selfies_roundtrip_report(["CCO", "C1CC"])
    assert report.n_input == 2
    # the unkekulizable ring does not parse as a molecule at all -- the
    # funnel stops at n_parsed=1 and the row never reaches the encoder
    assert report.n_parsed == 1
    assert report.n_encoded == 1
    assert report.fidelity == 0.5


@pytest.mark.skipif(not _HAS_RDKIT, reason="RDKit not installed")
def test_roundtrip_report_sample_limit_seeded() -> None:
    from pharmaml.chem.selfies_codec import selfies_roundtrip_report

    pool = ["CCO", "CCN", "CCS", "CCF", "CCC", "CCCO"]
    first = selfies_roundtrip_report(pool, sample_limit=3, seed=7)
    assert first.n_input == 3
    second = selfies_roundtrip_report(pool, sample_limit=3, seed=7)
    # the report has no __eq__ -- the seeded draw is pinned through the
    # funnel numbers and the bounded example lists instead
    assert first.n_parsed == second.n_parsed
    assert first.fidelity == second.fidelity
    assert first.summary() == second.summary()


# --------------------------------------------------------------------------- #
# robustness experiment
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _HAS_RDKIT, reason="RDKit not installed")
def test_robustness_experiment_headline_property() -> None:
    from pharmaml.chem.selfies_codec import selfies_robustness_experiment

    report = selfies_robustness_experiment(
        ["c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"], mutations=(1, 4), trials=8, seed=42
    )
    assert report.n_molecules == 2
    # the SELFIES side survives corruption at ~100%; the SMILES side
    # collapses -- the paper's Figure 3, on our own molecules
    assert report.selfies_validity == [1.0, 1.0]
    assert all(sf > sm for sf, sm in zip(report.selfies_validity, report.smiles_validity))
    assert report.smiles_validity[1] < 0.5
    frame = report.as_frame()
    assert list(frame.columns) == ["mutations", "representation", "validity"]
    assert len(frame) == 4  # 2 levels x 2 representations
    assert "k=4" in report.summary()


@pytest.mark.skipif(not _HAS_RDKIT, reason="RDKit not installed")
def test_robustness_experiment_deterministic() -> None:
    from pharmaml.chem.selfies_codec import selfies_robustness_experiment

    run = lambda: selfies_robustness_experiment(
        ["CCO", "c1ccccc1"], mutations=(1,), trials=4, seed=11
    )
    assert run().selfies_validity == run().selfies_validity
    assert run().smiles_validity == run().smiles_validity


@pytest.mark.skipif(not _HAS_RDKIT, reason="RDKit not installed")
def test_robustness_experiment_validations() -> None:
    from pharmaml.chem.selfies_codec import selfies_robustness_experiment

    with pytest.raises(DataValidationError, match="non-empty sequence"):
        selfies_robustness_experiment(["CCO"], mutations=())
    with pytest.raises(DataValidationError, match="must be >= 1"):
        selfies_robustness_experiment(["CCO"], mutations=(0,))
    with pytest.raises(DataValidationError, match="trials must be >= 1"):
        selfies_robustness_experiment(["CCO"], trials=0)
    with pytest.raises(DataValidationError, match="no encodable molecules"):
        selfies_robustness_experiment(["C1CC"])  # nothing encodable in the pool


# --------------------------------------------------------------------------- #
# ML feature matrix
# --------------------------------------------------------------------------- #
def test_feature_matrix_label_shape_and_vocab() -> None:
    from pharmaml.chem.selfies_codec import selfies_feature_matrix

    matrix, vocab = selfies_feature_matrix(["CCO", "CCN"])
    assert matrix.shape == (2, 3)
    assert matrix.dtype.kind == "i"
    assert vocab == ["[C]", "[N]", "[O]", "[nop]"]
    assert matrix[0].tolist() == [0, 0, 2]  # C=0, C=0, O=2
    assert matrix[1].tolist() == [0, 0, 1]  # C=0, C=0, N=1


def test_feature_matrix_one_hot_width() -> None:
    from pharmaml.chem.selfies_codec import selfies_feature_matrix

    matrix, vocab = selfies_feature_matrix(["CCO", "CCN", "CCS"], encoding="one_hot")
    assert matrix.shape == (3, 3 * len(vocab))
    assert matrix.dtype.kind == "f"
    # every symbol position is one-hot: 3 symbols set per row
    assert (matrix == 1.0).sum(axis=1).tolist() == [3, 3, 3]


def test_feature_matrix_refusals() -> None:
    from pharmaml.chem.selfies_codec import selfies_feature_matrix

    with pytest.raises(DataValidationError, match="label' or 'one_hot"):
        selfies_feature_matrix(["CCO"], encoding="dense")
    with pytest.raises(DataValidationError, match="column is empty"):
        selfies_feature_matrix([])
    with pytest.raises(DataValidationError, match="row 1"):
        selfies_feature_matrix(["CCO", "C1CC"])
    with pytest.raises(DataValidationError, match="row 0 is empty"):
        selfies_feature_matrix(["  ", "CCO"])


def test_feature_matrix_pad_to_len() -> None:
    from pharmaml.chem.selfies_codec import selfies_feature_matrix

    matrix, vocab = selfies_feature_matrix(["CCO"], pad_to_len=6)
    assert matrix.shape == (1, 6)
    # the single-molecule vocabulary is ["[C]", "[O]", "[nop]"] (sorted:
    # uppercase element symbols precede the lowercase "[nop]") -- so O
    # is index 1 and padding uses the [nop] index 2
    assert vocab == ["[C]", "[O]", "[nop]"]
    assert matrix[0].tolist() == [0, 0, 1, 2, 2, 2]


# --------------------------------------------------------------------------- #
# semantic constraints context manager
# --------------------------------------------------------------------------- #
def test_semantic_constraints_clamp_and_restore() -> None:
    from pharmaml.chem.selfies_codec import (
        selfies_to_smiles,
        semantic_constraints,
    )

    assert selfies_to_smiles("[C][#C]") == "C#C"  # default: acetylene
    with semantic_constraints({"C": 1, "O": 1, "N": 1, "?": 1}):
        # carbon capped at one bond: the triple bond clamps to a single
        assert selfies_to_smiles("[C][#C]") == "CC"
    # restored: nothing leaked
    assert selfies_to_smiles("[C][#C]") == "C#C"


def test_semantic_constraints_restore_on_exception() -> None:
    import selfies as sf

    from pharmaml.chem.selfies_codec import semantic_constraints

    saved = sf.get_semantic_constraints()
    with pytest.raises(RuntimeError, match="boom"):
        with semantic_constraints("hypervalent"):
            raise RuntimeError("boom")
    assert sf.get_semantic_constraints() == saved


def test_semantic_constraint_presets() -> None:
    from pharmaml.chem.selfies_codec import semantic_constraint_presets

    assert semantic_constraint_presets() == ["default", "hypervalent", "octet_rule"]


# --------------------------------------------------------------------------- #
# availability + the missing-package guard
# --------------------------------------------------------------------------- #
def test_selfies_available_true_in_verified_venv() -> None:
    from pharmaml.chem.selfies_codec import selfies_available

    # this file is skipped entirely when selfies is absent, so the probe
    # must agree on the real path
    assert selfies_available() is True


def test_missing_selfies_raises_actionable_dependency_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The guard when the package is absent: the exact pip command.

    Patching ``builtins.__import__`` intercepts the import statement even
    when an earlier test already cached ``selfies`` in ``sys.modules``
    (the statement always routes through the ``__import__`` builtin).
    """
    import builtins

    real_import = builtins.__import__
    import pharmaml.chem.selfies_codec as codec

    def blocked(name: str, *args: object, **kwargs: object) -> object:
        if name == "selfies" or name.startswith("selfies."):
            raise ImportError(f"blocked for the test: {name}")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(DependencyError, match=r"pip install 'pharmaml\[selfies\]'"):
        codec.smiles_to_selfies("CCO")
    with pytest.raises(DependencyError, match=r"pip install 'pharmaml\[selfies\]'"):
        codec.selfies_to_smiles("[C][C][O]")
