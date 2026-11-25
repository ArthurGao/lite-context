"""What a call to lite-context did, in numbers (design §5.2 rule 5)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Action:
    """One change a step made to one message: `ref` is the message's key, or `#<position>` in
    the input when it has none; inserted messages have a ref of their own."""

    step: str
    ref: str
    what: str  # "truncated" | "masked" | "dropped" | "inserted" | a host step's own word
    tokens_saved: int


@dataclass(frozen=True)
class Report:
    """Token numbers are estimates from the counter, never vendor-billed counts (design §4)."""

    limit: int
    reserve_output: int
    fixed_tokens: int
    available: int = 0
    tokens_before: int = 0
    tokens_after: int = 0
    actions: tuple[Action, ...] = ()
    compaction_due: bool = False
    estimated: tuple[str, ...] = ()  # refs of messages holding opaque parts with no known tokens
