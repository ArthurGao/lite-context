"""Where a `Transcript`'s entries are kept (design §8): the store contract, the record it keeps,
and `InMemoryStore`. `PgStore` (extra `pg`) keeps the same contract in PostgreSQL."""

from __future__ import annotations

import asyncio
import copy
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, Protocol

from .codec import Kind
from .errors import SeqConflict, UnknownTranscript


@dataclass(frozen=True)
class Record:
    """What a store keeps: `data` is JSON-ready (`codec.item_to_record`)."""

    seq: int
    kind: Kind
    data: dict[str, Any]


class Store(Protocol):
    async def create(self, transcript: str) -> None:
        """Create the transcript; creating one that exists does nothing."""
        ...

    async def append(
        self, transcript: str, records: Sequence[Record], *, expected_seq: int
    ) -> None:
        """Append `records`, whose seqs must run on from `expected_seq`, all or none. Raises
        `SeqConflict` when the tail is not `expected_seq`, `UnknownTranscript`, `ValueError`
        for a batch that is empty or does not follow on."""
        ...

    async def read(self, transcript: str, *, after: int = 0) -> list[Record]: ...

    async def tail(self, transcript: str) -> int:
        """The last seq, 0 when nothing is appended yet."""
        ...

    async def aclose(self) -> None: ...


def check_batch(
    transcript: str, records: Sequence[Record], *, expected_seq: int, tail: int
) -> None:
    """What every store checks before it writes, under its lock."""
    if tail != expected_seq:
        raise SeqConflict(transcript, expected_seq, tail)
    if not records:
        raise ValueError("an append needs at least one record")
    seqs = [r.seq for r in records]
    if seqs != list(range(expected_seq + 1, expected_seq + 1 + len(records))):
        raise ValueError(f"the batch's seqs {seqs} do not run on from {expected_seq}")


class InMemoryStore:
    """A store in a dict, for tests and for hosts that keep a transcript for one process."""

    def __init__(self) -> None:
        self._logs: dict[str, list[Record]] = {}
        self._lock = asyncio.Lock()

    def _log(self, transcript: str) -> list[Record]:
        if transcript not in self._logs:
            raise UnknownTranscript(f"no transcript {transcript!r}")
        return self._logs[transcript]

    async def create(self, transcript: str) -> None:
        self._logs.setdefault(transcript, [])

    async def append(
        self, transcript: str, records: Sequence[Record], *, expected_seq: int
    ) -> None:
        async with self._lock:
            log = self._log(transcript)
            check_batch(transcript, records, expected_seq=expected_seq, tail=len(log))
            # Copies: what is stored never changes afterwards, whatever the caller does.
            log.extend(replace(r, data=copy.deepcopy(r.data)) for r in records)

    async def read(self, transcript: str, *, after: int = 0) -> list[Record]:
        return [replace(r, data=copy.deepcopy(r.data)) for r in self._log(transcript)[after:]]

    async def tail(self, transcript: str) -> int:
        return len(self._log(transcript))

    async def aclose(self) -> None:
        return None


__all__ = ["InMemoryStore", "Record", "Store", "check_batch"]
