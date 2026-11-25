"""The steps `fit()` runs (design §5.3), and the protocol a host's own step follows.

A step works on entries: a message with its `ref` (the caller's key, or `#<position>` in the input).
An *always* step runs on every call. A *cutting* step lists its units, oldest first, and applies the
first `n` of them; `fit()` picks `n` and records it as the step's cut (design §5.4).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Literal, Protocol

from .count import Counter, count_message, longest_prefix
from .report import Action
from .structure import Batch, batches, human_turns
from .types import Message, Opaque, Part, Text, ToolCall, ToolResult

# The fixed placeholder texts (design §5.2 rule 6).
TRUNCATED = "[cut: {n} more characters not shown]"
MASKED_RESULT = "[result cleared to save context; call {name} again if you need it]"
MASKED_OPAQUE = "[{kind} removed to save context]"
DROPPED = "[{n} earlier messages not shown]"

__all__ = [
    "DROPPED",
    "MASKED_OPAQUE",
    "MASKED_RESULT",
    "TRUNCATED",
    "Action",
    "AlwaysStep",
    "CuttingStep",
    "DropOldest",
    "Entry",
    "MaskOpaque",
    "MaskToolResults",
    "Step",
    "StepContext",
    "TruncateToolResults",
    "Unit",
]


@dataclass(frozen=True)
class Entry:
    ref: str
    message: Message


@dataclass(frozen=True)
class Unit:
    """One thing a cutting step can cut. `refs`: the messages it covers (they make up the cut's
    digest, design §5.4); `at`: where inside them, for the step's own use."""

    refs: tuple[str, ...]
    at: tuple[int, ...] = ()


@dataclass(frozen=True)
class StepContext:
    counter: Counter
    protected: frozenset[str]  # refs of pinned messages and of the last human turn's message


class AlwaysStep(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def always(self) -> Literal[True]: ...

    def apply(
        self, entries: Sequence[Entry], ctx: StepContext
    ) -> tuple[list[Entry], list[Action]]: ...


class CuttingStep(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def always(self) -> Literal[False]: ...

    def units(self, entries: Sequence[Entry], ctx: StepContext) -> list[Unit]: ...

    def apply(
        self, entries: Sequence[Entry], units: Sequence[Unit], ctx: StepContext
    ) -> tuple[list[Entry], list[Action]]: ...


Step = AlwaysStep | CuttingStep


def _saved(before: Message, after: Message, counter: Counter) -> int:
    return count_message(before, counter) - count_message(after, counter)


def _messages(entries: Sequence[Entry]) -> list[Message]:
    return [e.message for e in entries]


def _rewrite(
    entries: Sequence[Entry],
    step: str,
    what: str,
    ctx: StepContext,
    parts_of: Callable[[int, Entry], tuple[Part, ...]],
) -> tuple[list[Entry], list[Action]]:
    """Give each unprotected entry the parts `parts_of` returns; report each one that changed."""
    out: list[Entry] = []
    actions: list[Action] = []
    for i, e in enumerate(entries):
        parts = e.message.parts if e.ref in ctx.protected else parts_of(i, e)
        if parts == e.message.parts:
            out.append(e)
            continue
        message = replace(e.message, parts=parts)
        out.append(Entry(e.ref, message))
        actions.append(Action(step, e.ref, what, _saved(e.message, message, ctx.counter)))
    return out, actions


# --- always ---


@dataclass(frozen=True)
class TruncateToolResults:
    """Cuts each tool result text longer than `max_tokens` and says how much is gone. It gives the
    same result for the same message every time, so it never moves the prompt-cache prefix."""

    max_tokens: int
    keep: Literal["head", "head_tail"] = "head"
    name: str = "truncate_tool_results"
    always: Literal[True] = True

    def apply(self, entries: Sequence[Entry], ctx: StepContext) -> tuple[list[Entry], list[Action]]:
        def parts_of(i: int, e: Entry) -> tuple[Part, ...]:
            return tuple(self._part(p, ctx.counter) for p in e.message.parts)

        return _rewrite(entries, self.name, "truncated", ctx, parts_of)

    def _part(self, part: Part, counter: Counter) -> Part:
        """The limit is per result: its text parts count together. When it is over, they are
        joined (newline-separated, keeping the first one's place and extra) and cut as one."""
        if not isinstance(part, ToolResult):
            return part
        at = [i for i, c in enumerate(part.content) if isinstance(c, Text)]
        texts = [c for c in part.content if isinstance(c, Text)]
        whole = "\n".join(t.text for t in texts)
        if not at or counter.text(whole) <= self.max_tokens:
            return part
        cut = self._cut(whole, counter)
        if counter.text(cut) >= counter.text(whole):
            return part  # the marker would cost more than it removes
        content = [
            Text(cut, texts[0].extra) if i == at[0] else c
            for i, c in enumerate(part.content)
            if i == at[0] or not isinstance(c, Text)
        ]
        return replace(part, content=tuple(content))

    def _cut(self, text: str, counter: Counter) -> str:
        if self.keep == "head":
            head = longest_prefix(text, self.max_tokens, counter, from_end=False)
            return f"{head}\n{TRUNCATED.format(n=len(text) - len(head))}"
        head = longest_prefix(text, self.max_tokens // 2, counter, from_end=False)
        rest = text[len(head) :]  # the tail comes only from what the head left
        tail = longest_prefix(rest, self.max_tokens - self.max_tokens // 2, counter, from_end=True)
        gone = len(rest) - len(tail)
        return f"{head}\n{TRUNCATED.format(n=gone)}\n{tail}"


# --- cutting ---


def _touches(batch: Batch, entries: Sequence[Entry], protected: frozenset[str]) -> bool:
    return any(entries[i].ref in protected for i in (batch.call_at, *batch.result_at))


@dataclass(frozen=True)
class MaskToolResults:
    """Replaces the results of old batches with a placeholder; the calls stay, so the model can
    call again. The last `keep_last` batches, an unfinished batch and protected ones are kept."""

    keep_last: int = 3
    name: str = "mask_tool_results"
    always: Literal[False] = False

    def units(self, entries: Sequence[Entry], ctx: StepContext) -> list[Unit]:
        found = batches(_messages(entries))
        old = found[: max(len(found) - self.keep_last, 0)]
        return [
            Unit(tuple(entries[i].ref for i in (b.call_at, *b.result_at)), (b.call_at,))
            for b in old
            if b.complete and not _touches(b, entries, ctx.protected)
        ]

    def apply(
        self, entries: Sequence[Entry], units: Sequence[Unit], ctx: StepContext
    ) -> tuple[list[Entry], list[Action]]:
        names = {  # each unit points at its batch's call message
            c.id: c.name
            for u in units
            for c in entries[u.at[0]].message.parts
            if isinstance(c, ToolCall)
        }

        def parts_of(i: int, e: Entry) -> tuple[Part, ...]:
            return tuple(
                replace(p, content=(Text(MASKED_RESULT.format(name=names[p.call_id])),))
                if isinstance(p, ToolResult) and p.call_id in names
                else p
                for p in e.message.parts
            )

        return _rewrite(entries, self.name, "masked", ctx, parts_of)


_MASKABLE = ("image", "document")


@dataclass(frozen=True)
class MaskOpaque:
    """Replaces old images and documents (in messages and in tool results) with a placeholder.
    Never thinking blocks; never in protected messages; the last `keep_last` are kept."""

    keep_last: int = 3
    name: str = "mask_opaque"
    always: Literal[False] = False

    def units(self, entries: Sequence[Entry], ctx: StepContext) -> list[Unit]:
        """One unit per message holding images or documents, so a message is never half masked;
        the last `keep_last` such messages are kept."""
        found = [Unit((e.ref,)) for e in entries if any(_maskable(p) for p in _flat(e.message))]
        old = found[: max(len(found) - self.keep_last, 0)]
        return [u for u in old if u.refs[0] not in ctx.protected]

    def apply(
        self, entries: Sequence[Entry], units: Sequence[Unit], ctx: StepContext
    ) -> tuple[list[Entry], list[Action]]:
        refs = {u.refs[0] for u in units}

        def parts_of(i: int, e: Entry) -> tuple[Part, ...]:
            if e.ref not in refs:
                return e.message.parts
            return tuple(_mask_part(p) for p in e.message.parts)

        return _rewrite(entries, self.name, "masked", ctx, parts_of)


def _maskable(part: Part) -> bool:
    return isinstance(part, Opaque) and part.kind in _MASKABLE


def _flat(message: Message) -> list[Part]:
    """A message's parts, with tool results' content in place of the results."""
    return [c for p in message.parts for c in (p.content if isinstance(p, ToolResult) else (p,))]


def _mask_one(part: Part) -> Part:
    return (
        Text(MASKED_OPAQUE.format(kind=part.kind))
        if isinstance(part, Opaque) and _maskable(part)
        else part
    )


def _mask_part(part: Part) -> Part:
    if isinstance(part, ToolResult):
        content = tuple(_mask_one(c) for c in part.content)
        return replace(part, content=content)  # type: ignore[arg-type]
    return _mask_one(part)


@dataclass(frozen=True)
class DropOldest:
    """Drops whole old turns (design §3.1), keeping the last `keep_last_turns` and every turn that
    holds a protected message, and puts one placeholder where the first dropped message was."""

    keep_last_turns: int = 4
    name: str = "drop_oldest"
    always: Literal[False] = False

    def units(self, entries: Sequence[Entry], ctx: StepContext) -> list[Unit]:
        turns = human_turns(_messages(entries))
        old = turns[: max(len(turns) - self.keep_last_turns, 0)]
        found = [tuple(e.ref for e in entries[start:end]) for start, end in old]
        return [Unit(refs) for refs in found if not ctx.protected.intersection(refs)]

    def apply(
        self, entries: Sequence[Entry], units: Sequence[Unit], ctx: StepContext
    ) -> tuple[list[Entry], list[Action]]:
        gone = {r for u in units for r in u.refs}
        out: list[Entry] = []
        actions: list[Action] = []
        for e in entries:
            if e.ref not in gone:
                out.append(e)
                continue
            if not actions:
                text = DROPPED.format(n=len(gone))
                note = Entry(f"+{self.name}", Message("user", (Text(text),)))
                out.append(note)
                cost = count_message(note.message, ctx.counter)
                actions.append(Action(self.name, note.ref, "inserted", -cost))
            actions.append(
                Action(self.name, e.ref, "dropped", count_message(e.message, ctx.counter))
            )
        return out, actions
