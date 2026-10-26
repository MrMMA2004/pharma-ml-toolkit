"""Tests for :mod:`pharmaml.utils.io` -- the I/O foundation of the package."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pharmaml.exceptions import DataValidationError
from pharmaml.utils.io import ensure_directory, read_table, safe_filename, write_table


class TestReadTable:
    def test_csv_comma(self, tmp_path: Path) -> None:
        path = tmp_path / "t.csv"
        path.write_text("a,b\n1,2\n", encoding="utf-8")
        frame = read_table(path)
        assert frame.shape == (1, 2)
        assert frame["a"].tolist() == [1]

    def test_tsv_extension_reads_tab_separated(self, tmp_path: Path) -> None:
        path = tmp_path / "t.tsv"
        path.write_text("a\tb\n1\t2\n", encoding="utf-8")
        frame = read_table(path)
        assert frame.shape == (1, 2)

    def test_txt_with_pipe_delimiter_is_sniffed(self, tmp_path: Path) -> None:
        path = tmp_path / "t.txt"
        path.write_text("a|b\n1|2\n", encoding="utf-8")
        frame = read_table(path)
        assert frame.shape == (1, 2)

    def test_semicolon_csv_sniffed_not_one_giant_column(self, tmp_path: Path) -> None:
        """European-locale exports must not collapse into a single column."""
        path = tmp_path / "euro.csv"
        path.write_text("a;b;c\n1;2;3\n4;5;6\n", encoding="utf-8")
        frame = read_table(path)
        assert frame.shape == (2, 3)

    def test_column_headers_stripped(self, tmp_path: Path) -> None:
        path = tmp_path / "spaces.csv"
        path.write_text("Ligand , SMILES\nx, C\n", encoding="utf-8")
        frame = read_table(path)
        assert list(frame.columns) == ["Ligand", "SMILES"]

    def test_excel_sheet_default(self, tmp_path: Path) -> None:
        pytest.importorskip("openpyxl")
        path = tmp_path / "t.xlsx"
        pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_excel(path, index=False)
        frame = read_table(path)
        assert frame.shape == (2, 2)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(DataValidationError, match="not found"):
            read_table(tmp_path / "nope.csv")

    def test_unsupported_extension_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "t.parquet"
        path.write_text("x", encoding="utf-8")
        with pytest.raises(DataValidationError, match="Unsupported input format"):
            read_table(path)

    def test_malformed_csv_raises_clean_error(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.csv"
        path.write_text('a,b\n"x,2\n', encoding="utf-8")  # unclosed quote
        with pytest.raises(DataValidationError, match="Failed to parse"):
            read_table(path)

    def test_extra_fields_are_truncated_not_fatal(self, tmp_path: Path) -> None:
        """pandas silently truncates over-long rows -- read_table follows suit."""
        path = tmp_path / "extra.csv"
        path.write_text("a,b\n1,2,3,4,5\n", encoding="utf-8")
        frame = read_table(path)
        assert list(frame.columns) == ["a", "b"]
        assert frame.shape == (1, 2)

    def test_duplicate_headers_after_strip_raise_in_excel(self, tmp_path: Path) -> None:
        """Excel allows two 'Ligand' headers -- read_table must refuse them.

        CSV headers are auto-deduplicated by pandas itself (a,a -> a,a.1);
        the xlsx path is the one that can genuinely produce duplicates.
        """
        pytest.importorskip("openpyxl")
        from openpyxl import Workbook

        book = Workbook()
        sheet = book.active
        sheet.append(["Ligand", "Ligand "])  # differ only in trailing space
        sheet.append(["lig1", "lig2"])
        path = tmp_path / "dup.xlsx"
        book.save(path)
        with pytest.raises(DataValidationError, match="Duplicate column"):
            read_table(path)


class TestReadTableWithProgress:
    """(100x-144) The progressive reader -- parity with read_table by
    test, not by refactor: every well-formed file the plain reader
    handles must come back identical (values AND dtypes), small files
    keep the plain reader's behaviour byte for byte, and the ragged
    edge cases delegate to the sync verdict instead of inventing a
    different answer."""

    @staticmethod
    def _report_into(seen: list) -> object:
        def report(done: int, total: object, note: str) -> None:
            seen.append((done, total, note))

        return report

    def test_small_file_takes_the_plain_reader_without_reports(
        self, tmp_path: Path
    ) -> None:
        from pharmaml.utils.io import read_table_with_progress

        path = tmp_path / "tiny.csv"
        path.write_text("a,b\n1,2\n", encoding="utf-8")
        seen: list[tuple[int, object, str]] = []
        frame = read_table_with_progress(path, self._report_into(seen))
        assert frame.shape == (1, 2)
        assert seen == []  # under the 8 MiB threshold -- no reports

    def test_chunks_report_with_estimated_total(self, tmp_path: Path) -> None:
        from pharmaml.utils.io import read_table_with_progress

        path = tmp_path / "counted.csv"
        rows = ["ligand,pic50"] + [f"L{i:03d},{i:03d}.5" for i in range(25)]
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        seen: list[tuple[int, object, str]] = []
        frame = read_table_with_progress(
            path, self._report_into(seen), chunk_rows=10, size_threshold=1
        )
        assert len(frame) == 25
        # uniform line lengths -> the estimate is the true line count
        assert [entry[0] for entry in seen] == [10, 20, 25]
        assert all(entry[1] == 26 for entry in seen)  # header + 25 rows
        assert all(entry[2] == "" for entry in seen)

    def test_parity_with_read_table_across_delimiters_and_dtypes(
        self, tmp_path: Path
    ) -> None:
        import numpy as np
        from pandas.testing import assert_frame_equal

        from pharmaml.utils.io import read_table, read_table_with_progress

        # mixed int/float/NA/object columns over three delimiters -- the
        # chunked concat must match the plain read on VALUES and dtypes
        base = pd.DataFrame(
            {
                "n": [1, 2, 3, 4, 5, 6, 7, 8],
                "x": [1.5, np.nan, 2.5, 3.5, np.nan, 4.5, 5.5, 6.5],
                "s": ["a", "b", None, "d", "e", "f", "g", "h"],
            }
        )
        for suffix, sep in (("csv", ","), ("tsv", "\t"), ("txt", "|")):
            path = tmp_path / f"mixed.{suffix}"
            base.to_csv(path, sep=sep, index=False)
            seen: list[tuple[int, object, str]] = []
            frame = read_table_with_progress(
                path, self._report_into(seen), chunk_rows=3, size_threshold=1
            )
            assert_frame_equal(frame, read_table(path))
            assert [entry[0] for entry in seen] == [3, 6, 8]

    def test_header_only_file_round_trips_like_read_table(self, tmp_path: Path) -> None:
        from pharmaml.utils.io import read_table, read_table_with_progress

        path = tmp_path / "header_only.csv"
        path.write_text("a,b\n", encoding="utf-8")
        seen: list[tuple[int, object, str]] = []
        frame = read_table_with_progress(
            path, self._report_into(seen), chunk_rows=2, size_threshold=1
        )
        assert frame.shape == (0, 2)
        assert list(frame.columns) == ["a", "b"]
        assert read_table(path).shape == (0, 2)
        assert seen == []  # the empty chunk reports nothing

    def test_ragged_semicolon_file_matches_the_sync_verdict(
        self, tmp_path: Path
    ) -> None:
        # the auditor's parity probe: a sniffed ";" separator that
        # chokes on a ragged row must return the SAME mangled comma
        # frame read_table returns -- never a different answer
        from pharmaml.utils.io import read_table, read_table_with_progress

        path = tmp_path / "ragged.csv"
        path.write_text("a;b\n1;2\n3;4;5\n", encoding="utf-8")
        seen: list[tuple[int, object, str]] = []
        frame = read_table_with_progress(
            path, self._report_into(seen), chunk_rows=2, size_threshold=1
        )
        plain = read_table(path)
        assert list(frame.columns) == list(plain.columns) == ["a;b"]
        assert len(frame) == len(plain) == 2

    def test_truly_malformed_file_raises_the_clean_error(self, tmp_path: Path) -> None:
        # BOTH separators choke (unclosed quote) -- the sync ladder's
        # DataValidationError, unchanged
        from pharmaml.utils.io import read_table_with_progress

        path = tmp_path / "bad.csv"
        path.write_text('a,b\n"x,2\n', encoding="utf-8")
        with pytest.raises(DataValidationError, match="Failed to parse"):
            read_table_with_progress(path, lambda *a: None, size_threshold=1)

    def test_missing_file_and_excel_route_through_read_table(
        self, tmp_path: Path
    ) -> None:
        from pharmaml.utils.io import read_table_with_progress

        with pytest.raises(DataValidationError, match="not found"):
            read_table_with_progress(tmp_path / "nope.csv", lambda *a: None)
        pytest.importorskip("openpyxl")
        path = tmp_path / "t.xlsx"
        pd.DataFrame({"a": [1, 2]}).to_excel(path, index=False)
        seen: list[tuple[int, object, str]] = []
        frame = read_table_with_progress(path, self._report_into(seen))
        assert frame.shape == (2, 1)
        assert seen == []  # Excel always takes the plain reader

    def test_giant_first_line_degrades_to_open_ended_reports(
        self, tmp_path: Path
    ) -> None:
        # a first line longer than the 64 KiB sample -> no bytes-per-line
        # average -> no fake total: the reports carry None and the caller
        # renders an open-ended counter
        from pharmaml.utils.io import read_table_with_progress

        path = tmp_path / "giant.csv"
        path.write_text("a" * 70_000 + ",b\n" + "1,2\n" * 6, encoding="utf-8")
        seen: list[tuple[int, object, str]] = []
        frame = read_table_with_progress(
            path, self._report_into(seen), chunk_rows=4, size_threshold=1
        )
        assert len(frame) == 6
        assert seen and all(entry[1] is None for entry in seen)

    def test_the_estimator_is_honest_on_both_sides(self, tmp_path: Path) -> None:
        from pharmaml.utils.io import _estimate_csv_rows

        normal = tmp_path / "normal.csv"
        normal.write_text("a,b\n" + "1,2\n" * 6, encoding="utf-8")
        assert _estimate_csv_rows(normal, sample_bytes=64) == 7  # header + 6
        giant = tmp_path / "giant.csv"
        giant.write_text("a" * 100 + ",b\n" + "1,2\n" * 6, encoding="utf-8")
        # the sample holds no newline at all -- None, never a fake total
        assert _estimate_csv_rows(giant, sample_bytes=64) is None

    def test_duplicate_headers_after_strip_still_raise(self, tmp_path: Path) -> None:
        from pharmaml.utils.io import read_table_with_progress

        path = tmp_path / "dup.csv"
        path.write_text("a, a \n1,2\n3,4\n5,6\n7,8\n", encoding="utf-8")
        with pytest.raises(DataValidationError, match="Duplicate column"):
            read_table_with_progress(path, lambda *a: None, chunk_rows=2, size_threshold=1)

    def test_exported_from_the_package_root(self) -> None:
        import pharmaml.utils.io as io_mod

        assert "read_table_with_progress" in io_mod.__all__
        from pharmaml.utils import io as io_pkg

        assert callable(io_pkg.read_table_with_progress)


class TestWriteTable:
    def test_writes_csv_and_creates_parents(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "deep" / "out.csv"
        written = write_table(pd.DataFrame({"a": [1]}), path)
        assert written.is_file()
        assert written == path

    def test_writes_excel(self, tmp_path: Path) -> None:
        pytest.importorskip("openpyxl")
        path = tmp_path / "out.xlsx"
        written = write_table(pd.DataFrame({"a": [1]}), path)
        assert written.suffix == ".xlsx"

    def test_unknown_extension_falls_back_to_csv(self, tmp_path: Path, caplog) -> None:
        path = tmp_path / "out.dat"
        written = write_table(pd.DataFrame({"a": [1]}), path)
        assert written.suffix == ".csv"
        assert written.name == "out.csv"
        assert not path.exists()

    def test_returns_resolved_path(self, tmp_path: Path) -> None:
        written = write_table(pd.DataFrame({"a": [1]}), tmp_path / "x.csv")
        assert isinstance(written, Path)


class TestEnsureDirectory:
    def test_creates_nested(self, tmp_path: Path) -> None:
        target = tmp_path / "a" / "b" / "c"
        result = ensure_directory(target)
        assert result.is_dir()
        assert result == target

    def test_idempotent(self, tmp_path: Path) -> None:
        ensure_directory(tmp_path / "d")
        ensure_directory(tmp_path / "d")  # must not raise
        assert (tmp_path / "d").is_dir()


class TestSafeFilename:
    """The portability guards of safe_filename."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("ligand_07", "ligand_07"),
            ("ligand 7", "ligand_7"),
            # path separators vanish; interior dots are legal characters
            ("ligand/../../etc", "ligand_.._.._etc"),
            # apostrophe + space collapse into one separator; a trailing
            # unicode char is dropped entirely by the collapse step
            ("C'est un ligandé", "C_est_un_ligand"),
        ],
    )
    def test_basic_sanitisation(self, raw: str, expected: str) -> None:
        assert safe_filename(raw) == expected

    def test_unicode_ligand_name(self) -> None:
        # " é" is one unsafe run -> one separator; trailing separator is
        # collapsed away, so the name ends clean at "ligand".
        assert safe_filename("ligand é") == "ligand"

    def test_collapses_repeats(self) -> None:
        assert safe_filename("a  b") == "a_b"
        assert safe_filename("a__b") == "a_b"

    def test_empty_and_blank_fall_back_to_unnamed(self) -> None:
        assert safe_filename("") == "unnamed"
        assert safe_filename("   ") == "unnamed"
        assert safe_filename("///") == "unnamed"

    def test_bare_dots_fall_back_to_unnamed(self) -> None:
        assert safe_filename(".") == "unnamed"
        assert safe_filename("..") == "unnamed"
        assert safe_filename("...") == "unnamed"

    def test_trailing_dots_and_spaces_stripped(self) -> None:
        assert safe_filename("ligand.") == "ligand"
        assert safe_filename("ligand . ") == "ligand"

    @pytest.mark.parametrize(
        "reserved",
        ["CON", "PRN", "AUX", "NUL", "COM1", "COM9", "LPT1", "LPT9",
         "con", "aux", "com3"],
    )
    def test_windows_reserved_names_prefixed(self, reserved: str) -> None:
        result = safe_filename(reserved)
        assert result == f"_{reserved}"

    def test_windows_reserved_with_extension(self) -> None:
        assert safe_filename("CON.csv") == "_CON.csv"
        assert safe_filename("nul.dat") == "_nul.dat"

    def test_non_reserved_lookalikes_untouched(self) -> None:
        # Stems that merely *contain* reserved substrings stay as they are.
        assert safe_filename("console") == "console"
        assert safe_filename("my.connect") == "my.connect"
        assert safe_filename("lpt") == "lpt"  # bare stem, not LPT<n>

    def test_long_names_truncated_to_200(self) -> None:
        assert len(safe_filename("a" * 500)) == 200

    def test_custom_replacement(self) -> None:
        assert safe_filename("a b", replacement="-") == "a-b"
