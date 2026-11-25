"""Batches, turns and pairing (design §3.1)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .errors import BrokenPairing
from .types import Message, Text, ToolCall, ToolResult


@dataclass(frozen=True)
class Batch:
    """One assistant message's tool calls and the messages holding their results."""

    call_at: int
    result_at: tuple[int, ...]
    names: dict[str, str]  # call id -> tool name
    complete: bool


def batches(messages: Sequence[Message]) -> list[Batch]:
    """The batches of `messages`, oldest first. Raises `BrokenPairing` on a result without its
    call, a second result for one call, or calls left unanswered when anything other than their
    results comes next. Only the last batch may still be waiting for results."""
    found: list[Batch] = []
    names: dict[str, str] = {}
    waiting: set[str] = set()
    result_at: list[int] = []
    call_at = -1

    def close(complete: bool) -> None:
        if call_at >= 0:
            found.append(Batch(call_at, tuple(result_at), dict(names), complete))

    for i, m in enumerate(messages):
        results = [p for p in m.parts if isinstance(p, ToolResult)]
        if waiting and not results:
            raise BrokenPairing(f"message {i}: calls {sorted(waiting)} have no result")
        if m.role == "assistant":
            if results:
                raise BrokenPairing(f"message {i}: an assistant message holds tool results")
            close(True)
            call_at, result_at = -1, []
            calls = [p for p in m.parts if isinstance(p, ToolCall)]
            if calls:
                call_at, names = i, {c.id: c.name for c in calls}
                waiting = set(names)
            continue
        for r in results:
            if r.call_id not in waiting:
                why = "a second result" if r.call_id in names else "no call"
                raise BrokenPairing(f"message {i}: result for {r.call_id!r} has {why}")
            waiting.discard(r.call_id)
        if results:
            result_at.append(i)
    close(not waiting)
    return found


def _is_human(m: Message) -> bool:
    return (
        m.role == "user"
        and any(isinstance(p, Text) for p in m.parts)
        and not any(isinstance(p, ToolResult) for p in m.parts)
    )


def human_turns(messages: Sequence[Message]) -> list[tuple[int, int]]:
    """`[start, end)` of each turn: from a user message with text and no tool result to the next."""
    starts = [i for i, m in enumerate(messages) if _is_human(m)]
    return list(zip(starts, [*starts[1:], len(messages)], strict=False))  # [] when no starts


def last_human(messages: Sequence[Message]) -> int | None:
    turns = human_turns(messages)
    return turns[-1][0] if turns else None
