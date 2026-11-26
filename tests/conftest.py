"""Shared test setup. Tests never open a socket (AGENTS.md rule 10), except to the PostgreSQL
test database, and only in a test that asks for it (`pg_url`): libpq connects outside Python's
socket module, and the only Python call let through is the address lookup of that host."""

import contextlib
import os
import socket
from collections.abc import Callable, Iterator
from typing import Any
from urllib.parse import urlsplit

import pytest

# PG tests read their URL only from LITECONTEXT_TEST_DATABASE_URL and only ever use the schema
# below, which they drop and recreate. Without the variable they fail, not skip: run
# `uv run --env-file .env pytest`.
TEST_SCHEMA = "lite_context_test"
_DB_HOSTS: set[str] = set()  # filled by `pg_url` for the test that asked for it


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("a test tried to open a network connection")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket.socket, "sendto", refuse)  # UDP needs no connect()
    real_getaddrinfo = socket.getaddrinfo

    def lookup(host: Any, *args: Any, **kwargs: Any) -> Any:
        if host in _DB_HOSTS:
            return real_getaddrinfo(host, *args, **kwargs)
        raise AssertionError("a test tried to open a network connection")

    monkeypatch.setattr(socket, "getaddrinfo", lookup)  # a DNS lookup is a network call too


@pytest.fixture
def pg_url() -> Iterator[str]:
    url = os.environ.get("LITECONTEXT_TEST_DATABASE_URL")
    if not url:
        pytest.fail(
            "LITECONTEXT_TEST_DATABASE_URL is not set: PostgreSQL tests need it (see .env.example)"
        )
    host = urlsplit(url).hostname or ""
    _DB_HOSTS.add(host)
    yield url
    _DB_HOSTS.discard(host)


@contextlib.contextmanager
def fresh_schema(url: str, schema: str, *, migrated: bool = True) -> Iterator[str]:
    """`schema` dropped and (when `migrated`) migrated, dropped again afterwards."""
    import psycopg
    from psycopg import sql

    from lite_context.pg import migrate

    drop = sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(schema))
    with psycopg.connect(url, autocommit=True) as conn:
        conn.execute(drop)
        if migrated:
            migrate(conn, schema)
    try:
        yield schema
    finally:
        with psycopg.connect(url, autocommit=True) as conn:
            conn.execute(drop)


@pytest.fixture
def pg_schema(pg_url: str) -> Iterator[str]:
    """A freshly migrated, empty `lite_context_test` schema (dropped again afterwards)."""
    with fresh_schema(pg_url, TEST_SCHEMA) as schema:
        yield schema


@pytest.fixture
def pg_store_factory(pg_url: str, pg_schema: str) -> Callable[[], Any]:
    from lite_context.pg import PgStore

    # Connections open from the start, so concurrent appends really overlap in the database.
    return lambda: PgStore(pg_url, schema=pg_schema, min_size=4, max_size=8)
