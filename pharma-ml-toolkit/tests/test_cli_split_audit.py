"""End-to-end safety tests for the split overlap audit CLI."""
import json
import os
import subprocess
import sys

import pandas as pd
import pytest

from pharmaml.cli import main


def run(capsys, train, test, *options):
    code = main(["data", "audit-splits", "--partition", f"train={train}",
                 "--partition", f"test={test}", "--key", "id", *options])
    return code, capsys.readouterr()


def test_exact_text_ids_and_private_summary(tmp_path, capsys):
    train = tmp_path / "train.csv"
    test = tmp_path / "test.tsv"
    train.write_text('id,value\n001,x\nNA,x\nSECRET,x\nSECRET,x\n', encoding="utf-8")
    test.write_text('id\tvalue\n1\tx\nNA\tx\nSECRET\tx\n', encoding="utf-8")
    code, out = run(capsys, train, test)
    assert code == 3
    assert 'shared keys: 2' in out.out
    assert 'train: rows=4 duplicate_rows=1 overlap_rows=3' in out.out
    assert 'test: rows=3 duplicate_rows=0 overlap_rows=2' in out.out
    assert 'SECRET' not in out.out and 'SECRET' not in out.err


def test_composite_detail_export_and_no_clobber(tmp_path, capsys):
    train, test, details = (tmp_path / name for name in ('a.csv', 'b.csv', 'details.json'))
    train.write_text('id,site\n001,A\n001,B\n', encoding='utf-8')
    test.write_text('id,site\n001,B\n001,C\n', encoding='utf-8')
    code, _ = run(capsys, train, test, '--key', 'site', '--details-json', str(details))
    assert code == 3
    assert json.loads(details.read_text())['shared_keys'] == [
        {'key': ['001', 'B'], 'counts': {'train': 1, 'test': 1}}
    ]
    if os.name == 'posix':
        assert details.stat().st_mode & 0o077 == 0
    before = details.read_bytes()
    code, out = run(capsys, train, test, '--key', 'site', '--details-json', str(details))
    assert code == 1 and 'Cannot write' in out.err
    assert details.read_bytes() == before


def test_clean_excel_and_explicit_delimiter(tmp_path, capsys):
    train, test = tmp_path / 'train.xlsx', tmp_path / 'test.csv'
    pd.DataFrame({'id': ['001', 'NA']}).to_excel(train, index=False)
    test.write_text('id;value\n002;x\n', encoding='utf-8')
    code, out = run(capsys, train, test, '--delimiter', 'semicolon')
    assert code == 0 and 'shared keys: 0' in out.out


@pytest.mark.parametrize('contents, message', [
    ('id,id\na,b\n', 'duplicate column'),
    ('id,other\n,x\n', 'empty'),
    ('other\nx\n', 'missing key'),
    ('id,other\na,x,z\n', 'Cannot read'),
    ('id\n\n', 'inconsistent field'),
])
def test_bad_input_fails_closed(tmp_path, capsys, contents, message):
    train, test = tmp_path / 'a.csv', tmp_path / 'b.csv'
    train.write_text(contents, encoding='utf-8')
    test.write_text('id\nx\n', encoding='utf-8')
    code, out = run(capsys, train, test)
    assert code == 1 and message.lower() in out.err.lower()


def test_argument_validation_and_existing_input_preserved(tmp_path, capsys):
    path = tmp_path / 'data.csv'
    path.write_text('id\nx\n', encoding='utf-8')
    for args in [
        ['--partition', f'train={path}'],
        ['--partition', f'train={path}', '--partition', f'train={path}'],
        ['--partition', f'bad name={path}', '--partition', f'test={path}'],
        ['--partition', f'train={path}', '--partition', f'test={path}', '--key', 'id'],
    ]:
        assert main(['data', 'audit-splits', '--key', 'id', *args]) == 1
        capsys.readouterr()
    code, _ = run(capsys, path, path, '--details-json', str(path))
    assert code == 1 and path.read_text() == 'id\nx\n'


def test_process_exit_code(tmp_path):
    a, b = tmp_path / 'a.csv', tmp_path / 'b.csv'
    a.write_text('id\nx\n', encoding='utf-8')
    b.write_text('id\nx\n', encoding='utf-8')
    result = subprocess.run([sys.executable, '-m', 'pharmaml.cli', 'data', 'audit-splits',
                             '--partition', f'a={a}', '--partition', f'b={b}', '--key', 'id'],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 3
    assert 'shared keys: 1' in result.stdout


def test_duplicate_excel_headers_are_rejected(tmp_path, capsys):
    train, test = tmp_path / 'a.xlsx', tmp_path / 'b.csv'
    pd.DataFrame([['x', 'y']], columns=['id', 'id']).to_excel(train, index=False)
    test.write_text('id\nx\n', encoding='utf-8')
    code, out = run(capsys, train, test)
    assert code == 1 and 'duplicate column' in out.err


def test_argument_errors_have_distinct_exit_code():
    result = subprocess.run([sys.executable, '-m', 'pharmaml.cli', 'data', 'audit-splits'],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 2


def test_malformed_quoted_csv_is_rejected(tmp_path, capsys):
    a, b = tmp_path / 'a.csv', tmp_path / 'b.csv'
    a.write_text('id\n"bad\n', encoding='utf-8')
    b.write_text('id\nx\n', encoding='utf-8')
    code, out = run(capsys, a, b)
    assert code == 1 and 'Cannot read' in out.err
