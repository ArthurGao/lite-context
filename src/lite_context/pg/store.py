"""`PgStore`: transcripts in PostgreSQL (design §8, extra `pg`). The same contract as
`InMemoryStore`: whole batches at the expected tail, nothing ever rewritten (the database's
triggers refuse it). Patterned on lite-agent's `PgSessionStore`."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Sequence
from typing import Any

import psycopg
from psycopg import AsyncConnection, AsyncCursor, sql
from psycopg.types.json import Json
from psycopg_pool import AsyncConnectionPool

from ..errors import LiteContextError, StoreError, UnknownTranscript
from ..store import Record, check_batch
from .db import check_schema, connection_error, redact_conninfo

MIGRATE = (
    "the schema has no lite-context tables: apply the migrations first (`litecontext migrate`)"
)


class PgStore:
    """Opens its connection pool on first use; `aclose()` when done. `schema` must be migrated
    first (`litecontext migrate`). Every database failure comes out as `StoreError`."""

    def __init__(
        self,
        database_url: str,
        *,
        schema: str = "lite_context",
        min_size: int = 1,
        max_size: int = 4,
    ) -> None:
        self.schema = check_schema(schema)
        self._url = database_url
        self._sizes = (min_size, max_size)
        self._pool: AsyncConnectionPool[AsyncConnection[Any]] | None = None
        self._closed = False  # aclose() is final
        self._opening = asyncio.Lock()
        transcripts = sql.Identifier(schema, "transcripts")
        self._entries = sql.Identifier(schema, "entries")
        # Composed once; every query names its schema, so no search_path is involved.
        self._create = sql.SQL(
            "INSERT INTO {} (id) VALUES (%s) ON CONFLICT (id) DO NOTHING"
        ).format(transcripts)
        self._exists = sql.SQL("SELECT 1 FROM {} WHERE id = %s").format(transcripts)
        self._tail = sql.SQL(
            "SELECT coalesce(max(seq), 0) FROM {} WHERE transcript_id = %s"
        ).format(self._entries)
        self._insert = sql.SQL(
            "INSERT INTO {} (transcript_id, seq, kind, data) VALUES (%s, %s, %s, %s)"
        ).format(self._entries)
        self._select = sql.SQL(
            "SELECT seq, kind, data FROM {} WHERE transcript_id = %s AND seq > %s ORDER BY seq"
        ).format(self._entries)

    async def _ready(self) -> AsyncConnectionPool[AsyncConnection[Any]]:
        async with self._opening:
            if self._closed:
                raise StoreError("the transcript store is closed")
            if self._pool is None:
                self._pool = await self._open()
            return self._pool

    async def _open(self) -> AsyncConnectionPool[AsyncConnection[Any]]:
        """A new, checked pool. A failed attempt leaves nothing behind, so the next call tries
        again from scratch (a closed pool cannot be reopened)."""
        min_size, max_size = self._sizes
        pool: AsyncConnectionPool[AsyncConnection[Any]] = AsyncConnectionPool(
            self._url,
            min_size=min_size,
            max_size=max_size,
            open=False,
            kwargs={"autocommit": True},
        )
        # Up to the first successful query every failure can be about the connection: none of
        # them may pass psycopg's message on, which can quote the connection string.
        try:
            await pool.open(wait=True, timeout=5)
            async with pool.connection() as conn:
                found = await conn.execute(
                    "SELECT to_regclass(%s)", (self._entries.as_string(conn),)
                )
                row = await found.fetchone()
        except Exception as exc:
            await pool.close()
            raise connection_error(exc, self._url) from None
        if row is None or row[0] is None:
            await pool.close()
            raise StoreError(f"{MIGRATE} (schema {self.schema})")
        return pool

    @contextlib.asynccontextmanager
    async def _cursor(self, *, transaction: bool = False) -> AsyncIterator[AsyncCursor[Any]]:
        pool = await self._ready()
        try:
            async with pool.connection() as conn:
                block = conn.transaction() if transaction else contextlib.nullcontext()
                async with block, conn.cursor() as cur:
                    yield cur
        except LiteContextError:
            raise  # SeqConflict, UnknownTranscript, a batch refused by check_batch
        except psycopg.Error as exc:
            # The type and the SQLSTATE only: a server message can quote the data written.
            raise StoreError(
                f"PostgreSQL at {redact_conninfo(self._url)} refused: {type(exc).__name__}"
                f" ({exc.sqlstate})"
            ) from None

    async def aclose(self) -> None:
        async with self._opening:
            self._closed = True
            if self._pool is not None:
                await self._pool.close()

    async def create(self, transcript: str) -> None:
        async with self._cursor() as cur:
            await cur.execute(self._create, (transcript,))

    async def append(
        self, transcript: str, records: Sequence[Record], *, expected_seq: int
    ) -> None:
        async with self._cursor(transaction=True) as cur:
            # One writer per transcript at a time: the second waits here, then sees the new tail.
            tail = await self._tail_of(cur, transcript, lock=True)
            check_batch(transcript, records, expected_seq=expected_seq, tail=tail)
            await cur.executemany(
                self._insert, [(transcript, r.seq, r.kind, Json(r.data)) for r in records]
            )

    async def read(self, transcript: str, *, after: int = 0) -> list[Record]:
        async with self._cursor() as cur:
            await self._tail_of(cur, transcript)
            await cur.execute(self._select, (transcript, after))
            return [Record(seq, kind, data) for seq, kind, data in await cur.fetchall()]

    async def tail(self, transcript: str) -> int:
        async with self._cursor() as cur:
            return await self._tail_of(cur, transcript)

    async def _tail_of(self, cur: AsyncCursor[Any], transcript: str, *, lock: bool = False) -> int:
        """The transcript's last seq; `UnknownTranscript` when there is none. With `lock`, its
        row is held until the transaction ends."""
        await cur.execute(
            self._exists + sql.SQL(" FOR UPDATE") if lock else self._exists, (transcript,)
        )
        if await cur.fetchone() is None:
            raise UnknownTranscript(f"no transcript {transcript!r}")
        await cur.execute(self._tail, (transcript,))
        row = await cur.fetchone()
        assert row is not None  # coalesce(max(seq), 0) always gives a row
        return int(row[0])


__all__ = ["PgStore", "check_schema"]
