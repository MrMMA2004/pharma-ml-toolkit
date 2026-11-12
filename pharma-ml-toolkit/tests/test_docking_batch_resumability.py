"""The §13 exact-restore proofs for the resumable docking batch.

Part5-continuation round 10: ``dock_smiles_array(checkpoint_dir=...)``
is the fifth of the six pending §13 operation classes to earn registry
entry.  These tests are the evidence, driven by a DETERMINISTIC fake
engine (no real Vina / AutoDock4 binary runs here -- the §14 archive
audit already pins the real engine checksums).  The resume unit is the
LIGAND: the checkpoint is the per-ligand completion ledger, and
finished ligands are PRESERVED verbatim, never re-run -- a real
docking engine is stochastic by nature, so "preserve" (not
re-execute) is the honest exactness contract.  The fingerprint closes
the two gaps the R9 audit flagged for this round: the FULL smiles
list + names (a reordered or edited array is a different dataset and
refuses to resume) and the receptor PDBQT content checksum (an edited
receptor refuses too).  A cooperative stop preserves the partial
result frame on ``JobCancelled.partial_results``.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest

import pharmaml.docking.smiles_docking as smiles_mod
import pharmaml.docking.workflow as workflow_mod
from pharmaml.docking.smiles_docking import dock_smiles_array
from pharmaml.exceptions import JobCancelled
from pharmaml.utils.checkpoint_schema import (
    ENVELOPE_KEY,
    OPERATION_REGISTRY,
    OperationSpec,
    validate_checkpoint,
)

LIGANDS = [
    "CCO",  # ethanol
    "c1ccccc1C(=O)O",  # benzoic acid
    "CC(=O)Oc1ccccc1C(=O)O",  # aspirin
    "CN1CCC[C@H]1c1cccnc1",  # nicotine
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # ibuprofen
]


def _fake_engine(target: Path) -> Path:
    """The deterministic fake prep + engine (the §14-proof seam)."""
    engines = target / "engines"
    engines.mkdir(exist_ok=True)
    (engines / "vina.exe").write_bytes(b"")
    return engines


def _fake_prep(smiles: str, folder: str, name: str = "l", seed: int = 0) -> Path:
    p = Path(folder)
    p.mkdir(parents=True, exist_ok=True)
    lig = p / f"{name}.pdbqt"
    lig.write_text("ATOM  stub\n", encoding="utf-8")
    return lig


def _fake_vina(*args: object, **kwargs: object) -> object:
    out = Path(kwargs["out"])  # type: ignore[arg-type]
    score = -5.0 - (len(out.parent.name) % 7) * 0.5
    out.write_text(
        f"REMARK VINA RESULT:    {score:.1f}      0.000      0.000\n"
        "REMARK VINA RESULT:    -4.0      0.000      0.000\nEND\n",
        encoding="utf-8",
    )
    return workflow_mod.DockingRunResult(out.parent, True, out, 0.0)


def _receptor(tmp_path: Path) -> Path:
    receptor = tmp_path / "receptor.pdbqt"
    receptor.write_text("ROOT\n", encoding="utf-8")
    return receptor


def _dock(
    tmp_path: Path,
    out_name: str,
    *,
    smiles: list[str] | None = None,
    receptor: Path | None = None,
    checkpoint_dir: Path | None = None,
    checkpoint_every: int = 1,
    checkpoint_resume: bool = True,
    progress: object = None,
    names: list[str] | None = None,
    workers: int = 1,
) -> object:
    engines = _fake_engine(tmp_path)
    kwargs: dict[str, object] = {
        "engine": "vina",
        "engine_dir": engines,
        "checkpoint_dir": checkpoint_dir,
        "checkpoint_every": checkpoint_every,
        "checkpoint_resume": checkpoint_resume,
        "workers": workers,
    }
    if progress is not None:
        kwargs["progress"] = progress
    if names is not None:
        kwargs["names"] = names
    with mock.patch.object(smiles_mod, "smiles_to_pdbqt", _fake_prep), \
         mock.patch.object(smiles_mod, "run_vina", _fake_vina), \
         mock.patch.object(
             smiles_mod, "engine_argv", lambda n, d=None: ["/bin/true"]
         ):
        return dock_smiles_array(
            smiles if smiles is not None else LIGANDS,
            receptor if receptor is not None else _receptor(tmp_path),
            tmp_path / out_name,
            **kwargs,
        )


def _frames_equal(a: object, b: object) -> bool:
    return a.reset_index(drop=True).equals(b.reset_index(drop=True))  # type: ignore[attr-defined]


class TestDockingBatchResume:
    def test_resumed_batch_is_bit_exact(self, tmp_path: Path) -> None:
        """THE registry proof: a full fresh run (checkpoints written,
        resume suppressed) vs a resume of it in the SAME campaign root
        -- the resume re-docks NOTHING and the result frame is
        identical, verbatim rows included (folders and outputs point
        at the same root)."""
        home = tmp_path / "ckpt"
        fresh = _dock(tmp_path, "out", checkpoint_dir=home,
                      checkpoint_resume=False)
        assert len(fresh) == len(LIGANDS)  # type: ignore[arg-type]
        assert int(fresh["ok"].sum()) == len(LIGANDS)  # type: ignore[index]
        # the resume: count engine invocations through the fake
        calls: list[Path] = []
        real_fake_vina = _fake_vina

        def counting_vina(*args: object, **kwargs: object) -> object:
            calls.append(Path(kwargs["out"]))  # type: ignore[arg-type]
            return real_fake_vina(*args, **kwargs)

        engines = _fake_engine(tmp_path)
        with mock.patch.object(smiles_mod, "smiles_to_pdbqt", _fake_prep), \
             mock.patch.object(smiles_mod, "run_vina", counting_vina), \
             mock.patch.object(
                 smiles_mod, "engine_argv", lambda n, d=None: ["/bin/true"]
             ):
            resumed = dock_smiles_array(
                LIGANDS,
                _receptor(tmp_path),
                tmp_path / "out",  # the same campaign root
                engine="vina",
                engine_dir=engines,
                checkpoint_dir=home,
            )
        assert calls == []  # nothing re-docked
        assert _frames_equal(resumed, fresh)

    def test_partial_checkpoint_resumes_only_the_tail(
        self, tmp_path: Path
    ) -> None:
        """A checkpoint from a genuinely partial run: the resume docks
        only the remaining ligands, in the SAME campaign root (the
        folders are the durable artifacts -- the realistic resume),
        and still matches the fresh run."""
        home = tmp_path / "partial"
        with pytest.raises(JobCancelled):
            _dock(
                tmp_path,
                "out",
                checkpoint_dir=home,
                progress=lambda done, total, note: (
                    False if (str(note).startswith("docking") and done >= 3)
                    else None
                ),
            )
        steps = sorted(home.glob("docking-batch.step-*.json"))
        assert steps, "the partial run must have written checkpoints"
        stopped_at = max(
            int(s.name.split(".step-")[1].split(".")[0]) for s in steps
        )
        assert stopped_at >= 3
        calls: list[Path] = []
        real_fake_vina = _fake_vina

        def counting_vina(*args: object, **kwargs: object) -> object:
            calls.append(Path(kwargs["out"]))  # type: ignore[arg-type]
            return real_fake_vina(*args, **kwargs)

        engines = _fake_engine(tmp_path)
        with mock.patch.object(smiles_mod, "smiles_to_pdbqt", _fake_prep), \
             mock.patch.object(smiles_mod, "run_vina", counting_vina), \
             mock.patch.object(
                 smiles_mod, "engine_argv", lambda n, d=None: ["/bin/true"]
             ):
            report = dock_smiles_array(
                LIGANDS,
                _receptor(tmp_path),
                tmp_path / "out",  # the same campaign root
                engine="vina",
                engine_dir=engines,
                checkpoint_dir=home,
            )
        assert len(calls) == len(LIGANDS) - stopped_at  # tail only
        # the fresh twin runs in ITS OWN root: folder/output carry the
        # campaign path, so the comparison is on the path-independent
        # columns; the preserved rows point at the resumed root
        fresh = _dock(tmp_path, "out-twin")
        cols = [c for c in report.columns if c not in ("folder", "output")]
        assert report[cols].equals(fresh[cols])  # type: ignore[index]
        assert (
            report["folder"].str.startswith(str(tmp_path / "out"))  # type: ignore[index]
        ).all()

    def test_cancel_preserves_partial_frame_on_the_exception(
        self, tmp_path: Path
    ) -> None:
        """§13: preserve completed partial results after cancellation --
        the partial result frame rides JobCancelled (ranked, one row
        per ligand, the finished ligands scored)."""
        home = tmp_path / "stop"
        with pytest.raises(JobCancelled) as caught:
            _dock(
                tmp_path,
                "out",
                checkpoint_dir=home,
                progress=lambda done, total, note: (
                    False if (str(note).startswith("docking") and done >= 2)
                    else None
                ),
            )
        partial = caught.value.partial_results
        assert partial is not None
        assert len(partial) == len(LIGANDS)  # type: ignore[arg-type]
        assert int(partial["ok"].sum()) == 2  # type: ignore[index]
        # the unfinished ligands carry no score and no error yet
        pending_rows = partial[~partial["ok"]]  # type: ignore[index]
        assert (pending_rows["best_score"].isna()).all()  # type: ignore[index]
        assert (pending_rows["error"] == "").all()  # type: ignore[index]

    def test_stale_dataset_checkpoint_refuses_to_resume(
        self, tmp_path: Path
    ) -> None:
        """The R9-gap closure #1: the FULL smiles list (and the names)
        are fingerprinted -- an edited, extended, reordered or renamed
        dataset refuses to resume (exact restore or nothing)."""
        home = tmp_path / "stale-smiles"
        _dock(tmp_path, "out", checkpoint_dir=home, checkpoint_resume=False)
        variants: list[dict[str, object]] = [
            {"smiles": LIGANDS[:-1]},  # shortened
            {"smiles": LIGANDS + ["CCC"]},  # extended
            {"smiles": list(reversed(LIGANDS))},  # reordered
            {"names": [f"lig{i}" for i in range(len(LIGANDS))]},
        ]
        for variant in variants:
            n_expected = len(variant.get("smiles", LIGANDS))  # type: ignore[arg-type]
            calls: list[Path] = []

            def counting_vina(
                *args: object,
                _calls: list[Path] = calls,
                **kwargs: object,
            ) -> object:
                _calls.append(Path(kwargs["out"]))  # type: ignore[arg-type]
                return _fake_vina(*args, **kwargs)

            engines = _fake_engine(tmp_path)
            with mock.patch.object(
                    smiles_mod, "smiles_to_pdbqt", _fake_prep), \
                 mock.patch.object(smiles_mod, "run_vina", counting_vina), \
                 mock.patch.object(
                     smiles_mod, "engine_argv",
                     lambda n, d=None: ["/bin/true"]
                 ):
                dock_smiles_array(
                    variant.get("smiles", LIGANDS),  # type: ignore[arg-type]
                    _receptor(tmp_path),
                    tmp_path / "out-v",
                    engine="vina",
                    engine_dir=engines,
                    checkpoint_dir=home,
                    names=variant.get("names"),  # type: ignore[arg-type]
                )
            assert len(calls) == n_expected, (
                f"stale variant {variant} must re-dock everything"
            )

    def test_stale_receptor_checkpoint_refuses_to_resume(
        self, tmp_path: Path
    ) -> None:
        """The R9-gap closure #2: the receptor PDBQT content checksum
        is fingerprinted -- an edited receptor refuses to resume."""
        home = tmp_path / "stale-receptor"
        receptor = _receptor(tmp_path)
        _dock(
            tmp_path, "out", checkpoint_dir=home,
            checkpoint_resume=False, receptor=receptor,
        )
        receptor.write_text("ROOT\nATOM  edited\n", encoding="utf-8")
        calls: list[Path] = []
        real_fake_vina = _fake_vina

        def counting_vina(*args: object, **kwargs: object) -> object:
            calls.append(Path(kwargs["out"]))  # type: ignore[arg-type]
            return real_fake_vina(*args, **kwargs)

        engines = _fake_engine(tmp_path)
        with mock.patch.object(smiles_mod, "smiles_to_pdbqt", _fake_prep), \
             mock.patch.object(smiles_mod, "run_vina", counting_vina), \
             mock.patch.object(
                 smiles_mod, "engine_argv", lambda n, d=None: ["/bin/true"]
             ):
            dock_smiles_array(
                LIGANDS,
                receptor,
                tmp_path / "out2",
                engine="vina",
                engine_dir=engines,
                checkpoint_dir=home,
            )
        assert len(calls) == len(LIGANDS)  # everything re-docked

    def test_checkpoints_carry_the_v2_envelope(self, tmp_path: Path) -> None:
        """The §13 envelope rides every checkpoint: registered
        operation, resume_unit in finished ligands, validate green."""
        home = tmp_path / "envelope"
        _dock(tmp_path, "out", checkpoint_dir=home, checkpoint_resume=False)
        files = sorted(home.glob("docking-batch.step-*.json"))
        assert files
        for path in files:
            report = validate_checkpoint(
                path, manager_name="docking-batch"
            )
            assert report.ok, report.errors
            assert not report.legacy
            envelope = report.envelope
            assert envelope is not None
            assert envelope["operation"]["type"] == "docking_batch"
            assert envelope["operation"]["restore_exact"] is True
            assert (
                envelope["operation"]["partial_results_policy"]
                == "preserve"
            )
            record = json.loads(path.read_text(encoding="utf-8"))
            assert record[ENVELOPE_KEY]["resume_unit"] == record["step"]
            # the payload is the ledger -- per-ligand rows
            ledger = record["state"]["ledger"]
            assert ledger
            for key, entry in ledger.items():
                assert key == f"{entry['index']}:{entry['smiles']}"
                assert "best_score" in entry

    def test_registry_entry_pins_the_proof_test(self) -> None:
        """The registered entry points at THIS test (earning
        registration is the point -- the proof must be findable from
        the registry)."""
        entry = OPERATION_REGISTRY["docking_batch"]
        assert (
            entry.implementation
            == "pharmaml.docking.smiles_docking"
        )
        assert (
            "test_resumed_batch_is_bit_exact" in entry.exact_restore_proof
        )
        assert entry.unit_description == "ligands docked"
        # the pending frontier is preprocessing alone now
        with pytest.raises(Exception, match="preprocessing"):
            OperationSpec(operation_type="preprocessing", operation_id="x")

    def test_checkpoint_every_validation(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="checkpoint_every"):
            _dock(tmp_path, "out", checkpoint_dir=tmp_path,
                  checkpoint_every=0)

    def test_checkpoint_every_thins_the_saves(self, tmp_path: Path) -> None:
        """checkpoint_every=2 saves every second finished ligand; the
        rotation keeps the newest three -- and the resume (same
        campaign root) still preserves everything."""
        home = tmp_path / "thin"
        _dock(
            tmp_path, "out", checkpoint_dir=home,
            checkpoint_every=2, checkpoint_resume=False,
        )
        files = sorted(home.glob("docking-batch.step-*.json"))
        steps = [int(f.name.split(".step-")[1].split(".")[0]) for f in files]
        # saves at 2, 4 and the final 5 (the completion save) --
        # rotation keeps the newest three
        assert steps == [2, 4, 5]
        # the fresh twin runs in ITS OWN root (paths are part of the
        # preserved rows), then the resume compares against it on the
        # path-independent columns; the same-root resume keeps rows
        # verbatim including folders
        report = _dock(tmp_path, "out", checkpoint_dir=home)
        fresh = _dock(tmp_path, "out-twin")
        cols = [c for c in report.columns if c not in ("folder", "output")]
        assert report[cols].equals(fresh[cols])  # type: ignore[index]
        # and the preserved rows point at the SAME campaign root
        assert (
            report["folder"].str.startswith(str(tmp_path / "out"))  # type: ignore[index]
        ).all()

    def test_threaded_cancel_lands_at_a_clean_barrier(
        self, tmp_path: Path
    ) -> None:
        """workers=2: the stop request lands between waves -- the
        submitted subprocesses of the running wave always finish, and
        the partial frame still carries only finished, scored rows."""
        home = tmp_path / "threaded"
        with pytest.raises(JobCancelled) as caught:
            _dock(
                tmp_path,
                "out",
                checkpoint_dir=home,
                workers=2,
                progress=lambda done, total, note: (
                    False if (str(note).startswith("docking") and done >= 2)
                    else None
                ),
            )
        partial = caught.value.partial_results
        assert partial is not None
        # 2 finished (wave 1), then the stop lands before wave 2
        assert int(partial["ok"].sum()) == 2  # type: ignore[index]

    def test_preparation_failure_is_final_and_ledgered(
        self, tmp_path: Path
    ) -> None:
        """A ligand whose preparation fails is COMPLETE (error row):
        it rides the ledger, and a resume neither re-prepares it nor
        treats it as pending work."""
        home = tmp_path / "prep-fail"

        def failing_prep(
            smiles: str, folder: str, name: str = "l", seed: int = 0
        ) -> Path:
            if smiles == LIGANDS[2]:
                raise ValueError("embedding failed")
            return _fake_prep(smiles, folder, name, seed)

        engines = _fake_engine(tmp_path)
        with mock.patch.object(smiles_mod, "smiles_to_pdbqt", failing_prep), \
             mock.patch.object(smiles_mod, "run_vina", _fake_vina), \
             mock.patch.object(
                 smiles_mod, "engine_argv", lambda n, d=None: ["/bin/true"]
             ):
            dock_smiles_array(
                LIGANDS,
                _receptor(tmp_path),
                tmp_path / "out",
                engine="vina",
                engine_dir=engines,
                checkpoint_dir=home,
                checkpoint_resume=False,
            )
        record = json.loads(
            max(
                home.glob("docking-batch.step-*.json"),
                key=lambda p: p.name,
            ).read_text(encoding="utf-8")
        )
        failed = [
            e for e in record["state"]["ledger"].values()
            if "preparation failed" in e["error"]
        ]
        assert len(failed) == 1
        assert failed[0]["smiles"] == LIGANDS[2]
        # the resume preserves the failure verbatim (no re-prep):
        prep_calls: list[str] = []

        def counting_prep(
            smiles: str, folder: str, name: str = "l", seed: int = 0
        ) -> Path:
            prep_calls.append(smiles)
            return _fake_prep(smiles, folder, name, seed)

        engines = _fake_engine(tmp_path)
        with mock.patch.object(smiles_mod, "smiles_to_pdbqt", counting_prep), \
             mock.patch.object(smiles_mod, "run_vina", _fake_vina), \
             mock.patch.object(
                 smiles_mod, "engine_argv", lambda n, d=None: ["/bin/true"]
             ):
            resumed = dock_smiles_array(
                LIGANDS,
                _receptor(tmp_path),
                tmp_path / "out2",
                engine="vina",
                engine_dir=engines,
                checkpoint_dir=home,
            )
        assert LIGANDS[2] not in prep_calls  # never re-prepared
        failed_row = resumed[resumed["smiles"] == LIGANDS[2]]  # type: ignore[index]
        assert len(failed_row) == 1
        assert "preparation failed" in failed_row.iloc[0]["error"]
