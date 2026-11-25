"""Checks a host runs on its own step (design §5.3): the step goes through `fit()` on sample
conversations at several budgets, so the run-time rules of design §5.2 apply to it, and the
result must be deterministic and leave the input alone."""

from __future__ import annotations

import copy
from dataclasses import replace

from .count import Budget, CharEstimate, count_messages
from .errors import DoesNotFit
from .fit import fit
from .state import ContextState
from .steps import Step
from .structure import last_human
from .types import Message, Opaque, Text, ToolCall, ToolResult

COUNTER = CharEstimate(chars_per_token=1, per_message=0, image_tokens=100)


def sample_conversations() -> list[list[Message]]:
    """Histories with parallel batches, an unfinished batch, pinned messages, thinking blocks,
    and images: alone, two in one message, and one in each result of a parallel batch."""
    long = "x" * 300

    def turn(t: int, *, pinned: bool = False, parallel: bool = False) -> list[Message]:
        ids = [f"c{t}a", f"c{t}b"] if parallel else [f"c{t}"]
        return [
            Message("user", (Text(f"question {t}"),), key=f"h{t}", pinned=pinned),
            Message(
                "assistant",
                (
                    Opaque(kind="thinking", raw={"t": t}, text="thinking " * 5),
                    *(ToolCall(id=i, name="search", raw_args="{}") for i in ids),
                ),
                key=f"a{t}",
            ),
            Message(
                "user",
                tuple(
                    ToolResult(call_id=i, content=(Text(long), Opaque(kind="image", raw={"r": i})))
                    if parallel
                    else ToolResult(call_id=i, content=(Text(long),))
                    for i in ids
                ),
                key=f"r{t}",
            ),
            Message(
                "user",
                (
                    Text("see"),
                    *(Opaque(kind="image", raw={"img": t, "n": n}) for n in range(1 + t % 2)),
                ),
                key=f"i{t}",
            ),
            Message("assistant", (Text("answer"),), key=f"s{t}"),
        ]

    plain = [m for t in range(1, 6) for m in turn(t, parallel=t % 2 == 0)]
    pinned = [m for t in range(1, 6) for m in turn(t, pinned=t == 2)]
    unfinished = [*plain, Message("user", (Text("go on"),), key="h9")]
    unfinished += [
        Message("assistant", (ToolCall(id="z1", name="s", raw_args="{}"),), key="a9"),
    ]
    return [plain, pinned, unfinished]


def check_step(step: Step) -> int:
    """Raise if `step` breaks a rule of design §5.2 on any sample at any budget, or if its cuts do
    not survive a history that grows call by call with the state passed back (design §5.4).
    Returns how many runs changed something, so a test can tell the step was exercised."""
    acted = 0
    for messages in sample_conversations():
        total = count_messages(messages, COUNTER)
        for share in (0.9, 0.75, 0.6):
            acted += _grow(step, messages, Budget(int(total * share), 0))
        keyless = [replace(m, key=None) for m in messages]
        try:
            fit(
                keyless,
                budget=Budget(count_messages(messages, COUNTER) // 2, 0),
                steps=[step],
                counter=COUNTER,
            )
        except DoesNotFit:
            pass
        before = copy.deepcopy(messages)
        total = count_messages(messages, COUNTER)
        for limit in (total * 2, total, total * 3 // 4, total // 2):
            try:
                one = fit(messages, budget=Budget(limit, 0), steps=[step], counter=COUNTER)
                two = fit(messages, budget=Budget(limit, 0), steps=[step], counter=COUNTER)
            except DoesNotFit:
                continue
            if (one.messages, one.state, one.report) != (two.messages, two.state, two.report):
                raise AssertionError(f"{step.name}: two calls on one input gave different output")
            acted += bool(one.report.actions)
        if messages != before:
            raise AssertionError(f"{step.name}: the input was changed")
    return acted


def _grow(step: Step, messages: list[Message], budget: Budget) -> int:
    """Feed `messages` one message more per call, passing the state back. When the state did not
    move, the previous request must be a prefix of this one (the prompt cache holds), except the
    message that has just left protection (design §5.4 accepts that one change)."""
    state = ContextState()
    previous: list[Message] = []
    was_human: str | None = None
    acted = 0
    for end in range(1, len(messages) + 1):
        try:
            fitted = fit(messages[:end], budget=budget, steps=[step], state=state, counter=COUNTER)
        except DoesNotFit:
            break
        if fitted.state == state:
            moved = [a.key for a, b in zip(previous, fitted.messages, strict=False) if a != b]
            if set(moved) - {was_human}:
                raise AssertionError(
                    f"{step.name}: an earlier request changed while the state did not: {moved}"
                )
        acted += bool(fitted.report.actions)
        h = last_human(messages[:end])
        state, previous = fitted.state, fitted.messages
        was_human = messages[h].key if h is not None else None
    return acted
