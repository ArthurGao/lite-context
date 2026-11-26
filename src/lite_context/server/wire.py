"""Library types to and from JSON (design §9): the only place the two meet. No decisions here:
each function converts one way, and a value the library refuses keeps the library's error."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import Any

from .. import errors
from ..adapters import anthropic, openai
from ..codec import item_to_record, message_from_dict, message_to_dict, record_to_item
from ..compaction import CompactionPlan, accept_summary
from ..count import Budget
from ..fit import Fitted
from ..state import ContextState
from ..steps import DropOldest, MaskOpaque, MaskToolResults, Step, TruncateToolResults
from ..types import Message, with_keys
from .schemas import (
    AcceptIn,
    BudgetIn,
    DropOldestSpec,
    Format,
    MaskOpaqueSpec,
    MaskResultsSpec,
    StepSpec,
    TruncateSpec,
)


class Refused(Exception):
    """A request the library refused: its status and JSON body."""

    def __init__(self, status: int, error: str, detail: str, **extra: Any) -> None:
        super().__init__(detail)
        self.status = status
        self.body = {"error": error, "detail": detail, **extra}


_STATUS: dict[type[errors.LiteContextError], int] = {
    errors.StaleState: 409,
    errors.SeqConflict: 409,
    errors.UnknownTranscript: 404,
    errors.StoreError: 503,
}


def refused(exc: Exception) -> Refused:
    """The response for an error the library raised: its kind in snake case and its message.
    A `ValueError` (or a pattern `re` cannot compile) is input the library refused."""
    if isinstance(exc, errors.DoesNotFit):
        return Refused(422, "does_not_fit", str(exc), report=report(exc.report))
    if isinstance(exc, errors.SeqConflict):
        return Refused(409, "seq_conflict", str(exc), expected=exc.expected, actual=exc.actual)
    if isinstance(exc, errors.LiteContextError):
        name = "".join(f"_{c.lower()}" if c.isupper() else c for c in type(exc).__name__)
        return Refused(_STATUS.get(type(exc), 422), name.lstrip("_"), str(exc))
    return Refused(422, "invalid", str(exc))


def neutral(
    fmt: Format, native: Sequence[Any], keys: Sequence[str] | None = None
) -> tuple[list[Message], tuple[Mapping[str, object], ...], tuple[str, ...]]:
    """The messages as neutral types, with an OpenAI conversation's leading system messages:
    as given (to send back in front) and as text (fixed content for the budget)."""
    fixed: tuple[Mapping[str, object], ...] = ()
    texts: tuple[str, ...] = ()
    if fmt == "anthropic":
        messages = anthropic.to_messages(native)
    elif fmt == "openai":
        converted = openai.to_messages(native)
        messages, fixed, texts = converted.messages, converted.fixed, converted.fixed_texts()
    else:
        messages = [message_from_dict(m) for m in native]
    if keys is not None:
        messages = with_keys(messages, keys)
    return messages, fixed, texts


def native(
    fmt: Format, messages: Sequence[Message], fixed: Sequence[Mapping[str, object]] = ()
) -> list[Any]:
    if fmt == "anthropic":
        return anthropic.from_messages(messages)
    if fmt == "openai":
        return openai.from_messages(messages, fixed)
    return [message_to_dict(m) for m in messages]


def stored_message(fmt: Format, item: Any) -> list[Message]:
    """One message given to a transcript, as neutral messages (OpenAI may give several)."""
    messages, fixed, _ = neutral(fmt, [item])
    if fixed:
        raise ValueError("a system message is not stored in a transcript: pass it as fixed text")
    return messages


def state_in(data: Mapping[str, Any] | None) -> ContextState | None:
    if data is None:
        return None
    state = record_to_item("state", data)
    assert isinstance(state, ContextState)
    return state


def state_out(state: ContextState) -> Any:
    return item_to_record(state)[1]


def budget(b: BudgetIn) -> Budget:
    return Budget(b.limit, b.reserve_output)


def step(spec: StepSpec) -> Step:
    if isinstance(spec, TruncateSpec):
        return TruncateToolResults(max_tokens=spec.max_tokens)
    if isinstance(spec, MaskResultsSpec):
        return MaskToolResults(keep_last=spec.keep_last)
    if isinstance(spec, MaskOpaqueSpec):
        return MaskOpaque(keep_last=spec.keep_last)
    assert isinstance(spec, DropOldestSpec)
    return DropOldest(keep_last_turns=spec.keep_last_turns)


def report(r: Any) -> Any:
    return json.loads(json.dumps(asdict(r)))


def fitted(fmt: Format, f: Fitted, fixed: Sequence[Mapping[str, object]] = ()) -> dict[str, Any]:
    return {
        "messages": native(fmt, f.messages, fixed),
        "state": state_out(f.state),
        "report": report(f.report),
    }


def plan_out(fmt: Format, plan: CompactionPlan) -> dict[str, Any]:
    return {
        "system": plan.system,
        "prompt": native(fmt, plan.prompt),
        "covered": list(plan.covered),
        "mark": plan.mark,
        "keep_last_turns": plan.policy.keep_last_turns,
    }


def accepted(body: AcceptIn) -> ContextState:
    return accept_summary(
        body.reply,
        covered=body.covered,
        mark=body.mark,
        policy=body.policy(),
        truncated=body.truncated,
        provenance=body.provenance,
        now=body.now,
    )
