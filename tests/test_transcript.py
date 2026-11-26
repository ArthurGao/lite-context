"""`Transcript` (design §8): an append-only log of messages and states, and its projection."""

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import TypeVar

import pytest
from builders import keyless_turn

from lite_context import (
    Budget,
    ContextState,
    Message,
    Summary,
    SummaryPolicy,
    Text,
    fit,
)
from lite_context.errors import SeqConflict, UnknownTranscript
from lite_context.steps import MaskToolResults
from lite_context.store import InMemoryStore
from lite_context.testing import COUNTER
from lite_context.transcript import Entry, Transcript

T = TypeVar("T")
NOW = datetime(2026, 11, 26, tzinfo=UTC)
POLICY = SummaryPolicy(keep_last_turns=1, mark="m0")


def run(body: Callable[[], Awaitable[T]]) -> T:
    return asyncio.run(body())  # type: ignore[arg-type]


async def opened(store: InMemoryStore | None = None) -> Transcript:
    t = Transcript(store or InMemoryStore(), "t1")
    await t.create()
    return t


def test_messages_get_their_seq_as_key_and_come_back_with_it() -> None:
    async def body() -> list[Entry]:
        t = await opened()
        assert await t.append(keyless_turn(1)[:2], expected_seq=0) == 2
        assert await t.append(keyless_turn(1)[2:], expected_seq=2) == 4
        return await t.entries()

    entries = run(body)
    assert [e.seq for e in entries] == [1, 2, 3, 4]
    assert [e.item.key for e in entries if isinstance(e.item, Message)] == ["1", "2", "3", "4"]
    assert [e.item.parts for e in entries if isinstance(e.item, Message)] == [
        m.parts for m in keyless_turn(1)
    ]


def test_a_message_that_already_has_a_key_is_refused_and_nothing_is_written() -> None:
    async def body() -> list[Entry]:
        t = await opened()
        keyed = Message("user", (Text("x"),), key="mine")
        with pytest.raises(ValueError, match="key"):
            await t.append([Message("user", (Text("ok"),)), keyed], expected_seq=0)
        return await t.entries()

    assert run(body) == []


def test_a_stale_expected_seq_is_a_conflict() -> None:
    async def body() -> None:
        t = await opened()
        await t.append(keyless_turn(1), expected_seq=0)
        with pytest.raises(SeqConflict):
            await t.append(keyless_turn(2), expected_seq=0)
        assert await t.last_seq() == 4

    run(body)


def test_an_unknown_transcript_is_refused() -> None:
    async def body() -> None:
        t = Transcript(InMemoryStore(), "never-created")
        with pytest.raises(UnknownTranscript):
            await t.append(keyless_turn(1), expected_seq=0)

    run(body)


def test_the_projection_is_fit_on_the_messages_and_the_latest_state() -> None:
    steps = [MaskToolResults(keep_last=1)]
    budget = Budget(800, 0)  # 1,278 tokens; masking two results leaves 610

    async def body() -> None:
        t = await opened()
        seq = await t.append([*keyless_turn(1), *keyless_turn(2), *keyless_turn(3)], expected_seq=0)
        first = await t.project(budget=budget, steps=steps, counter=COUNTER)
        messages = [e.item for e in await t.entries() if isinstance(e.item, Message)]
        assert first == fit(messages, budget=budget, steps=steps, counter=COUNTER)
        assert first.state.cuts  # something was masked, and the cut is in the new state
        seq = await t.append([first.state], expected_seq=seq)
        seq = await t.append(keyless_turn(4, size=10), expected_seq=seq)
        again = await t.project(budget=budget, steps=steps, counter=COUNTER)
        # The state stored above is the one used: what was sent before is still a prefix.
        assert again.messages[: len(first.messages)] == first.messages
        assert again == fit(
            [e.item for e in await t.entries() if isinstance(e.item, Message)],
            budget=budget,
            steps=steps,
            state=first.state,
            counter=COUNTER,
        )

    run(body)


def test_compact_appends_one_state_with_the_summary_and_the_projection_shows_it() -> None:
    calls: list[str] = []

    async def summarize(system: str, prompt: list[Message]) -> Summary:
        calls.append(system)
        return Summary("they asked three times")

    async def body() -> None:
        t = await opened()
        seq = await t.append([*keyless_turn(1), *keyless_turn(2), *keyless_turn(3)], expected_seq=0)
        # An earlier state first, so the projection below must take the latest one.
        masked = await t.project(budget=Budget(800, 0), steps=[MaskToolResults(1)], counter=COUNTER)
        seq = await t.append([masked.state], expected_seq=seq)
        at = await t.compact(summarize=summarize, policy=POLICY, now=NOW)
        assert at == seq + 1
        last = (await t.entries())[-1].item
        assert isinstance(last, ContextState) and last.compaction is not None
        assert last.compaction.covers_through == "8"  # turns 1-2; the last turn is kept
        fitted = await t.project(budget=Budget(10_000, 0), counter=COUNTER)
        assert "they asked three times" in str(fitted.messages[0].parts[0])
        assert [m.key for m in fitted.messages[1:]] == ["9", "10", "11", "12"]
        # Nothing new to cover: no call, nothing appended.
        assert await t.compact(summarize=summarize, policy=POLICY, now=NOW) is None
        assert len(calls) == 1 and await t.last_seq() == at

    run(body)


def test_a_state_entry_round_trips_through_the_store() -> None:
    async def body() -> None:
        t = await opened()
        await t.append(keyless_turn(1), expected_seq=0)
        state = ContextState()
        fitted = await t.project(budget=Budget(200, 0), steps=[MaskToolResults(0)], counter=COUNTER)
        assert fitted.state != state
        await t.append([fitted.state], expected_seq=4)
        assert (await t.entries())[-1].item == fitted.state

    run(body)
