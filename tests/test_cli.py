"""`litecontext`: the command line (design §8; `serve` arrives with C6)."""

import pytest
from conftest import fresh_schema

from lite_context.cli import main
from lite_context.pg.db import migration_files

ALL = ", ".join(v for v, _ in migration_files())


@pytest.fixture(autouse=True)
def _never_a_real_server(monkeypatch: pytest.MonkeyPatch) -> None:
    """`serve` must never start uvicorn from a test, not even when a check under test is broken:
    the tests that expect it to run replace this stub with their own."""
    import uvicorn

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("a test started a real server")

    monkeypatch.setattr(uvicorn, "run", refuse)


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


def test_serve_refuses_an_address_that_is_not_loopback(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["serve", "--host", "0.0.0.0"]) == 2
    assert "loopback only" in capsys.readouterr().err


def test_serve_runs_the_app_on_the_given_loopback_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import uvicorn

    seen: dict[str, object] = {}

    def fake_run(app: object, *, host: str, port: int) -> None:
        seen.update(app=app, host=host, port=port)

    monkeypatch.setattr(uvicorn, "run", fake_run)
    monkeypatch.delenv("LITECONTEXT_DATABASE_URL", raising=False)
    monkeypatch.setenv("LITECONTEXT_MAX_BODY", "1000")
    assert main(["serve", "--port", "8799"]) == 0
    assert (seen["host"], seen["port"]) == ("127.0.0.1", 8799)
    assert getattr(seen["app"], "limit", None) == 1000


def test_serve_reads_host_and_port_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    import uvicorn

    seen: dict[str, object] = {}
    monkeypatch.setattr(
        uvicorn, "run", lambda app, *, host, port: seen.update(host=host, port=port)
    )
    monkeypatch.setenv("LITECONTEXT_HOST", "::1")
    monkeypatch.setenv("LITECONTEXT_PORT", "8751")
    assert main(["serve"]) == 0
    assert seen == {"host": "::1", "port": 8751}


@pytest.mark.parametrize(
    "name, value",
    [
        ("LITECONTEXT_PORT", "abc"),
        ("LITECONTEXT_PORT", "0"),
        ("LITECONTEXT_MAX_BODY", "1MB"),
        ("LITECONTEXT_MAX_BODY", "0"),
    ],
)
def test_a_bad_setting_for_serve_is_named_not_a_traceback(
    name: str, value: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(name, value)
    assert main(["serve"]) == 2
    err = capsys.readouterr().err
    assert name in err
    assert "--max_body" not in err and "--max-body" not in err  # there is no such flag


def test_a_bad_port_setting_does_not_stop_the_other_commands(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("LITECONTEXT_PORT", "abc")
    monkeypatch.delenv("LITECONTEXT_DATABASE_URL", raising=False)
    assert main(["pending"]) == 2
    assert "LITECONTEXT_DATABASE_URL is not set" in capsys.readouterr().err
