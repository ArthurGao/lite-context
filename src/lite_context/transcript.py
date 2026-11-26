"""`Transcript`: an append-only log of messages and `ContextState`s for hosts with no storage
of their own (design §8).

Each entry has a seq, 1, 2, 3, ... per transcript; a message's key is its seq, so the cuts in a
stored state keep pointing at the same messages forever. Appends name the tail they expect and
are all or nothing; nothing is ever changed or removed. `project()` is `fit()` over the messages
with the latest state, so the same entries always give the same output.

The store keeps JSON-ready records (`lite_context.codec`), so `InMemoryStore` and `PgStore`
(extra `pg`) behave alike, down to refusing a value JSON would not give back.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from typing import TypeAlias

from .codec import item_to_record, record_to_item
from .compaction import Summary, SummaryPolicy, acompact
from .count import Budget, Counter
from .fit import Fitted, fit
from .state import ContextState
from .steps import Step
from .store import Record, Store
from .types import Message

Item: TypeAlias = Message | ContextState


@dataclass(frozen=True)
class Entry:
    seq: int
    item: Item


class Transcript:
    """One transcript in a store. Holds no cache: every call reads the store, so several
    processes may share a transcript, with `expected_seq` deciding who appends first."""

    def __init__(self, store: Store, id: str) -> None:
        self.store = store
        self.id = id

    async def create(self) -> None:
        await self.store.create(self.id)

    async def last_seq(self) -> int:
        return await self.store.tail(self.id)

    async def append(self, items: Sequence[Item], *, expected_seq: int) -> int:
        """Append messages and states after `expected_seq`; returns the new last seq. A message
        gets its seq as its key, so it must come without one."""
        records: list[Record] = []
        for seq, item in enumerate(items, start=expected_seq + 1):
            if isinstance(item, Message):
                if item.key is not None:
                    raise ValueError(
                        f"a transcript keys its own messages; this one has {item.key!r}"
                    )
                item = replace(item, key=str(seq))
            records.append(Record(seq, *item_to_record(item)))
        await self.store.append(self.id, records, expected_seq=expected_seq)
        return expected_seq + len(records)

    async def entries(self) -> list[Entry]:
        return [
            Entry(r.seq, record_to_item(r.kind, r.data)) for r in await self.store.read(self.id)
        ]

    async def project(
        self,
        *,
        budget: Budget,
        steps: Sequence[Step] = (),
        fixed: Sequence[str | int] = (),
        counter: Counter | None = None,
        high: float = 0.9,
        low: float = 0.7,
    ) -> Fitted:
        """`fit()` over every message, with the latest stored state. Store `fitted.state` with
        `append` when it differs, so the next call cuts in the same places."""
        messages, state = _split(await self.entries())
        return fit(
            messages,
            budget=budget,
            steps=steps,
            fixed=fixed,
            state=state,
            counter=counter,
            high=high,
            low=low,
        )

    async def compact(
        self,
        *,
        summarize: Callable[[str, list[Message]], Awaitable[Summary]],
        policy: SummaryPolicy | None = None,
        now: datetime | None = None,
    ) -> int | None:
        """Summarise what is due (`acompact`, with the caller's model) and append the new state;
        returns its seq, or None when nothing new can be covered. Raises `SeqConflict` when
        someone appended while the summary was being written: run it again."""
        entries = await self.entries()
        tail = entries[-1].seq if entries else 0
        messages, state = _split(entries)
        new = await acompact(messages, summarize=summarize, state=state, policy=policy, now=now)
        if new is None:
            return None
        return await self.append([new], expected_seq=tail)


def _split(entries: Sequence[Entry]) -> tuple[list[Message], ContextState | None]:
    messages = [e.item for e in entries if isinstance(e.item, Message)]
    states = [e.item for e in entries if isinstance(e.item, ContextState)]
    return messages, (states[-1] if states else None)


__all__ = ["Entry", "Item", "Transcript"]
