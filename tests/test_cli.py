"""`litecontext`: the command line (design §8; `serve` arrives with C6)."""

import pytest
from conftest import fresh_schema

from lite_context.cli import main
from lite_context.pg.db import migration_files

ALL = ", ".join(v for v, _ in migration_files())


def test_migrate_applies_then_says_up_to_date(
    pg_url: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("LITECONTEXT_DATABASE_URL", pg_url)
    with fresh_schema(pg_url, "lite_context_test_cli", migrated=False) as schema:
        assert main(["pending", "--schema", schema]) == 0
        assert capsys.readouterr().out == f"pending in {schema}: {ALL}\n"
        assert main(["migrate", "--schema", schema]) == 0
        assert capsys.readouterr().out == f"applied {ALL} to {schema}\n"
        assert main(["migrate", "--schema", schema]) == 0
        assert capsys.readouterr().out == f"{schema} is up to date\n"
        assert main(["pending", "--schema", schema]) == 0
        assert capsys.readouterr().out == f"{schema} is up to date\n"


def test_without_a_database_url_it_says_which_variable(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("LITECONTEXT_DATABASE_URL", raising=False)
    assert main(["migrate"]) == 2
    assert "LITECONTEXT_DATABASE_URL is not set" in capsys.readouterr().err


def test_a_connection_failure_never_shows_the_password(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("LITECONTEXT_DATABASE_URL", "postgresql://u:hunter2-secret@127.0.0.1:1/x")
    assert main(["migrate"]) == 1
    err = capsys.readouterr().err
    assert "hunter2-secret" not in err and "cannot connect" in err


def test_a_schema_not_our_own_is_refused(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("LITECONTEXT_DATABASE_URL", "postgresql://u@127.0.0.1:1/x")
    assert main(["migrate", "--schema", "public"]) == 2
    assert "schema must be lite_context" in capsys.readouterr().err
