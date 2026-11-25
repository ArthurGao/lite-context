"""`lite_context.testing.check_step` runs a host's step through the rules (design §5.3)."""

from collections.abc import Sequence
from typing import Literal

import pytest

from lite_context import UnreportedChange
from lite_context.steps import (
    Action,
    DropOldest,
    Entry,
    MaskOpaque,
    MaskToolResults,
    Step,
    StepContext,
    TruncateToolResults,
)
from lite_context.testing import check_step


@pytest.mark.parametrize(
    "step",
    [TruncateToolResults(50), MaskToolResults(keep_last=1), MaskOpaque(keep_last=1), DropOldest(1)],
)
def test_the_built_in_steps_pass_and_are_exercised(step: Step) -> None:
    assert check_step(step) >= 2


class Quiet:
    """Shortens every assistant answer without saying so."""

    name = "quiet"
    always: Literal[True] = True

    def apply(self, entries: Sequence[Entry], ctx: StepContext) -> tuple[list[Entry], list[Action]]:
        from dataclasses import replace

        from lite_context import Text

        out = [
            Entry(e.ref, replace(e.message, parts=(Text("a"),)))
            if e.message.role == "assistant" and e.message.parts == (Text("answer"),)
            else e
            for e in entries
        ]
        return out, []


def test_a_host_step_that_hides_a_change_fails() -> None:
    with pytest.raises(UnreportedChange):
        check_step(Quiet())


class NewestFirst(MaskToolResults):
    """Lists its units newest first; `fit()` puts them in input order itself."""

    def units(self, entries: Sequence[Entry], ctx: StepContext) -> list:  # type: ignore[type-arg]
        return list(reversed(super().units(entries, ctx)))


def test_the_order_a_step_lists_its_units_in_does_not_matter() -> None:
    from lite_context import Budget, DoesNotFit, Fitted, fit
    from lite_context.count import count_messages
    from lite_context.testing import COUNTER, sample_conversations

    def outcome(msgs: list, step: Step, limit: int) -> Fitted | None:  # type: ignore[type-arg]
        try:
            return fit(msgs, budget=Budget(limit, 0), steps=[step], counter=COUNTER)
        except DoesNotFit:
            return None

    compared = 0
    for msgs in sample_conversations():
        total = count_messages(msgs, COUNTER)
        for share in (0.9, 0.75, 0.6):
            limit = int(total * share)
            a, b = outcome(msgs, MaskToolResults(1), limit), outcome(msgs, NewestFirst(1), limit)
            assert (a and (a.messages, a.state)) == (b and (b.messages, b.state))
            compared += a is not None and bool(a.report.actions)
    assert compared >= 3


class Ago(MaskToolResults):
    """Says how far back a result was: right on one call, but it rewrites an old message on
    every call after, without its cut moving."""

    def apply(self, entries, units, ctx):  # type: ignore[no-untyped-def]
        from dataclasses import replace

        from lite_context import Text, ToolResult

        out, actions = super().apply(entries, units, ctx)
        touched = {a.ref for a in actions}
        later = [
            Entry(
                e.ref,
                replace(
                    e.message,
                    parts=(
                        replace(
                            e.message.parts[0], content=(Text(f"{len(entries) - i} messages ago"),)
                        ),
                        *e.message.parts[1:],
                    ),
                ),
            )
            if e.ref in touched and isinstance(e.message.parts[0], ToolResult)
            else e
            for i, e in enumerate(out)
        ]
        return later, actions


def test_a_step_whose_output_drifts_as_the_history_grows_fails() -> None:
    with pytest.raises(AssertionError, match="earlier request changed"):
        check_step(Ago(keep_last=1))
