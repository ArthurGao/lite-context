"""PostgreSQL only (design §8, C5 acceptance): the database refuses rewrites, migrations are
idempotent, errors never show a password, and a transcript projects the same in memory, in
PostgreSQL and when read back by another process."""

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import psycopg
import pytest
from builders import keyless_turn
from psycopg import sql

from lite_context import Budget, ContextState, Message, Summary, SummaryPolicy, Text
from lite_context.codec import message_to_dict
from lite_context.errors import StoreError
from lite_context.pg import PgStore, migrate, pending
from lite_context.pg.db import migration_files
from lite_context.steps import MaskToolResults
from lite_context.store import InMemoryStore, Record, Store
from lite_context.testing import COUNTER
from lite_context.transcript import Transcript


def test_the_database_refuses_to_change_or_remove_an_entry(pg_url: str, pg_schema: str) -> None:
    async def write() -> None:
        store = PgStore(pg_url, schema=pg_schema)
        await store.create("t1")
        t = Transcript(store, "t1")
        await t.append([Message("user", (Text("x"),))], expected_seq=0)
        await store.aclose()

    asyncio.run(write())
    table = sql.Identifier(pg_schema, "entries")
    for statement in (
        sql.SQL("UPDATE {} SET kind = 'state'").format(table),
        sql.SQL("DELETE FROM {}").format(table),
        sql.SQL("TRUNCATE {}").format(table),
    ):
        with (
            psycopg.connect(pg_url, autocommit=True) as conn,
            pytest.raises(psycopg.errors.RaiseException, match="append-only"),
        ):
            conn.execute(statement)
    with psycopg.connect(pg_url, autocommit=True) as conn:
        assert conn.execute(sql.SQL("SELECT count(*) FROM {}").format(table)).fetchone() == (1,)


def test_migrating_again_applies_nothing(pg_url: str, pg_schema: str) -> None:
    with psycopg.connect(pg_url, autocommit=True) as conn:
        assert migrate(conn, pg_schema) == []
        assert pending(conn, pg_schema) == []
        assert pending(conn, "lite_context_never_made") == [v for v, _ in migration_files()]


async def first_call(url: str, schema: str) -> None:
    store = PgStore(url, schema=schema)
    try:
        await store.create("t1")
    finally:
        await store.aclose()


def test_a_bad_url_is_a_store_error_without_the_password() -> None:
    bad = "postgresql://someone:hunter2-secret@127.0.0.1:1/nowhere"
    with pytest.raises(StoreError) as caught:
        asyncio.run(first_call(bad, "lite_context"))
    assert "hunter2-secret" not in str(caught.value) and "***" in str(caught.value)


def test_an_unmigrated_schema_says_to_migrate(pg_url: str) -> None:
    with pytest.raises(StoreError, match="litecontext migrate"):
        asyncio.run(first_call(pg_url, "lite_context_never_made"))


def test_a_schema_that_is_not_our_own_is_refused() -> None:
    for schema in ("public", "lite_agent", "lite_context; drop", "Lite_Context"):
        with pytest.raises(ValueError, match="schema"):
            PgStore("postgresql://x@localhost/x", schema=schema)


# --- the same projection everywhere (C5 acceptance) ---

POLICY = SummaryPolicy(keep_last_turns=1, mark="m0")
KEEP = 1
STEPS = [MaskToolResults(keep_last=KEEP)]
BUDGET = Budget(900, 0)


async def converse(store: Store) -> list[str]:
    """A scripted conversation that grows, stores each new state and summarises once; returns
    every projection, as JSON, in order."""

    async def summarize(system: str, prompt: list[Message]) -> Summary:
        return Summary("the first turns, summarised")

    t = Transcript(store, "talk")
    await t.create()
    seq, seen, stored = 0, [], ContextState()
    for n in range(1, 7):
        seq = await t.append(keyless_turn(n, size=300), expected_seq=seq)
        fitted = await t.project(budget=BUDGET, steps=STEPS, counter=COUNTER)
        seen.append(projection(fitted.messages))
        if fitted.state != stored:
            seq = await t.append([fitted.state], expected_seq=seq)
            stored = fitted.state
        if n == 4:
            at = await t.compact(summarize=summarize, policy=POLICY)
            assert at is not None
            last = (await t.entries())[-1].item
            assert isinstance(last, ContextState)
            seq, stored = at, last
    return seen


def projection(messages: list[Message]) -> str:
    return json.dumps([message_to_dict(m) for m in messages], sort_keys=True)


def test_memory_and_postgresql_project_the_same(pg_url: str, pg_schema: str) -> None:
    async def both() -> tuple[list[str], list[str]]:
        in_memory = await converse(InMemoryStore())
        store = PgStore(pg_url, schema=pg_schema)
        try:
            return in_memory, await converse(store)
        finally:
            await store.aclose()

    in_memory, in_pg = asyncio.run(both())
    assert in_pg == in_memory
    assert len(set(in_memory)) > 2  # the projections did change along the way (masking, summary)


READ_BACK = """
import asyncio, json, sys
from lite_context.codec import message_to_dict
from lite_context.pg import PgStore
from lite_context.steps import MaskToolResults
from lite_context.testing import COUNTER
from lite_context import Budget
from lite_context.transcript import Transcript

async def main():
    url, schema, limit, keep = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    store = PgStore(url, schema=schema)
    try:
        fitted = await Transcript(store, "talk").project(
            budget=Budget(limit, 0), steps=[MaskToolResults(keep_last=keep)], counter=COUNTER)
    finally:
        await store.aclose()
    print(json.dumps([message_to_dict(m) for m in fitted.messages], sort_keys=True))

asyncio.run(main())
"""


def test_another_process_reads_the_transcript_back_and_projects_the_same(
    pg_url: str, pg_schema: str, tmp_path: Path
) -> None:
    async def write() -> str:
        store = PgStore(pg_url, schema=pg_schema)
        try:
            return (await converse(store))[-1]
        finally:
            await store.aclose()

    here = asyncio.run(write())
    script = tmp_path / "read_back.py"
    script.write_text(READ_BACK)
    out = subprocess.run(
        [sys.executable, "-I", str(script), pg_url, pg_schema, str(BUDGET.limit), str(KEEP)],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    assert out.stdout.strip() == here


def test_a_failed_first_connection_does_not_break_the_store_for_good() -> None:
    async def twice() -> list[str]:
        store = PgStore("postgresql://u@127.0.0.1:1/x", schema="lite_context")
        seen = []
        for _ in range(2):
            with pytest.raises(StoreError) as caught:
                await store.create("t1")
            seen.append(str(caught.value))
        await store.aclose()
        return seen

    first, second = asyncio.run(twice())
    assert first == second  # the second attempt connects again, not "the pool is closed"
    assert "PoolClosed" not in second


def test_a_database_error_after_connecting_is_a_store_error(pg_url: str, pg_schema: str) -> None:
    async def bad_record() -> None:
        store = PgStore(pg_url, schema=pg_schema)
        try:
            await store.create("t1")
            # Below the codec, which never writes this kind: the database's own refusal (its
            # CHECK on `kind`) must still come out as a StoreError, never as psycopg's.
            await store.append("t1", [Record(1, "bogus", {})], expected_seq=0)  # type: ignore[arg-type]
        finally:
            await store.aclose()

    with pytest.raises(StoreError, match="PostgreSQL"):
        asyncio.run(bad_record())


def test_migrations_refuse_a_schema_not_our_own(pg_url: str) -> None:
    with psycopg.connect(pg_url, autocommit=True) as conn:
        for call in (migrate, pending):
            with pytest.raises(ValueError, match="schema"):
                call(conn, "public")
