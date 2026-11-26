"""Summaries (design §6): planning, accepting, storing in the state and applying in `fit()`."""

import asyncio
import copy
from datetime import UTC, datetime

import pytest
from builders import COUNTER, calls, human, results, say

from lite_context import (
    Budget,
    Compaction,
    ContextState,
    EmptySummary,
    Message,
    StaleState,
    Summary,
    Text,
    TruncatedSummary,
    acompact,
    fit,
    plan_compaction,
)
from lite_context.compaction import ASK, END, SUMMARY, CompactionPlan, SummaryPolicy
from lite_context.state import digest as covered_digest
from lite_context.steps import MaskToolResults

NOW = datetime(2026, 11, 26, tzinfo=UTC)
POLICY = SummaryPolicy(keep_last_turns=1, mark="m0")


def turn(t: int) -> list[Message]:
    return [
        human(f"h{t}", f"question {t}"),
        calls(f"a{t}", f"c{t}"),
        results(f"r{t}", (f"c{t}", f"result {t}")),
        say(f"s{t}", f"answer {t}"),
    ]


def history(turns: int) -> list[Message]:
    return [m for t in range(1, turns + 1) for m in turn(t)]


def prompt_text(plan: CompactionPlan | None) -> str:
    assert plan is not None
    part = plan.prompt[0].parts[0]
    assert isinstance(part, Text)
    return part.text


def keys(msgs: list[Message]) -> list[str | None]:
    return [m.key for m in msgs]


# --- planning ---


def test_a_plan_covers_every_turn_but_the_kept_ones_and_renders_them() -> None:
    plan = plan_compaction(history(3), policy=POLICY)
    assert plan is not None
    assert plan.covered == ("h1", "a1", "r1", "s1", "h2", "a2", "r2", "s2")
    text = prompt_text(plan)
    assert "--- m0 user ---\nquestion 1" in text
    assert "(called search with {}; call id c2)" in text
    assert "--- m0 tool result for call id c2 ---\nresult 2" in text
    assert "question 3" not in text
    assert text.endswith(END.format(mark="m0"))
    assert plan.system == ASK.format(mark="m0")


def test_a_pinned_prefix_stays_out_and_a_later_pinned_message_stops_the_cut() -> None:
    msgs = [human("task", "the task", pinned=True), *history(4)]
    msgs[9] = human("h3", "question 3", pinned=True)  # the start of turn 3
    plan = plan_compaction(msgs, policy=POLICY)
    assert plan is not None
    assert plan.covered == ("h1", "a1", "r1", "s1", "h2", "a2", "r2", "s2")  # after the prefix


def test_nothing_to_cover_gives_no_plan() -> None:
    assert plan_compaction(history(1), policy=POLICY) is None


def test_compaction_needs_keys() -> None:
    msgs = [Message(m.role, m.parts) for m in history(3)]
    with pytest.raises(ValueError, match="key"):
        plan_compaction(msgs, policy=POLICY)


# --- accepting ---


def test_accepting_a_summary_gives_a_state_with_it_and_no_cuts() -> None:
    plan = plan_compaction(history(3), policy=POLICY)
    assert plan is not None
    state = plan.accept("they asked twice", provenance={"model": "m"}, now=NOW)
    c = state.compaction
    assert c is not None
    assert (c.covers_through, c.text) == ("s2", SUMMARY.format(summary="they asked twice"))
    assert c.covered_digest == covered_digest(["h1", "a1", "r1", "s1", "h2", "a2", "r2", "s2"])
    assert (c.provenance, c.created_at) == ({"model": "m"}, NOW.isoformat())
    assert state.cuts == {}


def test_an_empty_or_truncated_summary_is_refused() -> None:
    plan = plan_compaction(history(3), policy=POLICY)
    assert plan is not None
    with pytest.raises(EmptySummary):
        plan.accept("  \n", now=NOW)
    with pytest.raises(TruncatedSummary):
        plan.accept("half a sum", truncated=True, now=NOW)


def test_header_lines_the_model_copied_are_removed() -> None:
    plan = plan_compaction(history(3), policy=SummaryPolicy(keep_last_turns=1, mark="0" * 24))
    assert plan is not None
    state = plan.accept(f"--- {'0' * 24} user ---\nkept line", now=NOW)
    assert state.compaction is not None
    assert state.compaction.text == SUMMARY.format(summary="kept line")


# --- applying ---


def compacted(msgs: list[Message], policy: SummaryPolicy = POLICY) -> ContextState:
    plan = plan_compaction(msgs, policy=policy)
    assert plan is not None
    return plan.accept("S", now=NOW)


def test_fit_shows_the_summary_in_place_of_what_it_covers_and_reports_it() -> None:
    msgs = [human("task", "the task", pinned=True), *history(3)]
    before = copy.deepcopy(msgs)
    state = compacted(msgs)
    fitted = fit(msgs, budget=Budget(10_000, 0), state=state, counter=COUNTER)
    assert msgs == before
    summary = Message("user", (Text(SUMMARY.format(summary="S")),))
    assert fitted.messages == [msgs[0], summary, *msgs[9:]]
    acts = [(a.ref, a.what) for a in fitted.report.actions]
    assert acts[0] == ("+compaction", "inserted")
    assert acts[1:] == [(k, "summarised") for k in keys(msgs[1:9])]
    assert fitted.state.compaction == state.compaction


def test_a_later_compaction_takes_in_the_earlier_summary() -> None:
    msgs = history(3)
    first = compacted(msgs)
    grown = [*msgs, *turn(4)]
    plan = plan_compaction(grown, state=first, policy=POLICY)
    assert plan is not None
    text = prompt_text(plan)
    assert "--- m0 earlier summary ---\nS\n" in text
    assert "question 1" not in text and "question 3" in text
    second = plan.accept("S2", now=NOW)
    assert second.compaction is not None
    assert plan.covered == tuple(k for k in keys(grown[:12]) if k)  # from the start again
    assert second.compaction.covers_through == "s3"
    fitted = fit(grown, budget=Budget(10_000, 0), state=second, counter=COUNTER)
    assert keys(fitted.messages) == [None, "h4", "a4", "r4", "s4"]


def test_nothing_new_to_cover_after_a_compaction_gives_no_plan() -> None:
    msgs = history(3)
    assert plan_compaction(msgs, state=compacted(msgs), policy=POLICY) is None


@pytest.mark.parametrize(
    "change",
    ["drop-covered-end", "insert-before", "rekey-before", "boundary"],
)
def test_a_summary_that_no_longer_matches_the_history_is_stale(change: str) -> None:
    msgs = history(3)
    state = compacted(msgs)
    if change == "drop-covered-end":
        msgs = [m for m in msgs if m.key != "s2"]
    elif change == "insert-before":
        msgs = [human("h0", "zero"), say("s0", "z"), *msgs]
    elif change == "rekey-before":
        msgs = [Message(m.role, m.parts, key="H1") if m.key == "h1" else m for m in msgs]
    else:  # the summary would now end inside a turn
        msgs = [*msgs[:8], say("x", "more"), *msgs[8:]]
    with pytest.raises(StaleState):
        fit(msgs, budget=Budget(10_000, 0), state=state, counter=COUNTER)


def test_cuts_are_measured_after_the_summary_and_start_again_after_a_new_one() -> None:
    msgs = [
        m
        for t in range(1, 7)
        for m in (
            human(f"h{t}", f"question {t}"),
            calls(f"a{t}", f"c{t}"),
            results(f"r{t}", (f"c{t}", "x" * 200)),
            say(f"s{t}", f"answer {t}"),
        )
    ]
    state = compacted(msgs, SummaryPolicy(keep_last_turns=4, mark="m0"))
    step = MaskToolResults(keep_last=1)
    fitted = fit(msgs, budget=Budget(1200, 0), steps=[step], state=state, counter=COUNTER)
    assert fitted.state.compaction == state.compaction
    masked = [a.ref for a in fitted.report.actions if a.what == "masked"]
    assert masked == ["r3", "r4", "r5"]  # turns 1-2 are in the summary
    assert fitted.state.cuts["mask_tool_results"].through == "r5"
    again = fit(msgs, budget=Budget(1200, 0), steps=[step], state=fitted.state, counter=COUNTER)
    assert again.messages == fitted.messages
    newer = compacted(msgs, SummaryPolicy(keep_last_turns=2, mark="m0"))
    assert newer.cuts == {}


def test_state_with_a_compaction_survives_json() -> None:
    state = compacted(history(3))
    assert ContextState.from_json(state.to_json()) == state


# --- acompact ---


def test_acompact_plans_calls_the_callers_model_and_accepts() -> None:
    seen: list[list[Message]] = []

    async def summarize(system: str, messages: list[Message]) -> Summary:
        seen.append(messages)
        assert "m0" in system
        return Summary("all of it", provenance={"role": "compact"})

    state = asyncio.run(acompact(history(3), summarize=summarize, policy=POLICY, now=NOW))
    assert state is not None and state.compaction is not None
    assert state.compaction.text == SUMMARY.format(summary="all of it")
    assert len(seen) == 1


def test_compaction_is_a_frozen_record() -> None:
    c = compacted(history(3)).compaction
    assert isinstance(c, Compaction)
    with pytest.raises(AttributeError):
        c.text = "other"  # type: ignore[misc]


def test_a_summary_with_keyless_messages_is_stale() -> None:
    state = compacted(history(3))
    keyless = [Message(m.role, m.parts) for m in history(3)]
    with pytest.raises(StaleState, match="no keys"):
        fit(keyless, budget=Budget(10_000, 0), state=state, counter=COUNTER)


# --- regressions from the C1b review ---


def test_keep_last_turns_must_keep_at_least_the_current_turn() -> None:
    for bad in (0, -1):
        with pytest.raises(ValueError, match="keep_last_turns"):
            SummaryPolicy(keep_last_turns=bad)


def test_a_history_without_human_turns_gives_no_plan() -> None:
    msgs = [say("s0", "hello"), calls("a1", "c1"), results("r1", ("c1", "x")), say("s1", "done")]
    assert plan_compaction(msgs, policy=POLICY) is None


def test_the_plan_never_covers_an_unfinished_batch_or_the_last_turn() -> None:
    msgs = [*history(2), human("h3", "q3"), calls("a3", "c3")]
    plan = plan_compaction(msgs, policy=POLICY)
    assert plan is not None
    assert plan.covers_through == "s2"
    state = plan.accept("S", now=NOW)
    grown = [*msgs, results("r3", ("c3", "x")), say("s3", "ok")]
    assert (
        fit(grown, budget=Budget(10_000, 0), state=state, counter=COUNTER).messages[1:] == grown[8:]
    )


def test_provenance_is_stored_as_json_gives_it_back() -> None:
    plan = plan_compaction(history(3), policy=POLICY)
    assert plan is not None
    state = plan.accept("S", provenance={"usage": (1, 2)}, now=NOW)
    assert state.compaction is not None
    assert state.compaction.provenance == {"usage": [1, 2]}
    assert ContextState.from_json(state.to_json()) == state
    with pytest.raises(ValueError, match="provenance"):
        plan.accept("S", provenance={"at": NOW}, now=NOW)


def test_copied_headers_are_removed_whatever_the_mark() -> None:
    plan = plan_compaction(history(3), policy=POLICY)  # mark "m0", not hex
    assert plan is not None
    state = plan.accept("--- m0 user ---\nkept\n--- m0 end of the transcript ---", now=NOW)
    assert state.compaction is not None
    assert state.compaction.text == SUMMARY.format(summary="kept")


def test_the_earlier_summary_goes_back_to_the_model_without_its_preamble() -> None:
    msgs = history(3)
    first = compacted(msgs)
    plan = plan_compaction([*msgs, *turn(4)], state=first, policy=POLICY)
    text = prompt_text(plan)
    assert "--- m0 earlier summary ---\nS\n" in text
    assert SUMMARY.format(summary="") not in text


def test_fewer_turns_than_kept_gives_no_plan() -> None:
    assert plan_compaction(history(2), policy=SummaryPolicy(keep_last_turns=4)) is None


def test_accepting_from_the_plan_s_parts_equals_accepting_the_plan() -> None:
    from lite_context.compaction import accept_summary

    plan = plan_compaction(history(3), policy=POLICY)
    assert plan is not None
    by_parts = accept_summary(
        "S", covered=plan.covered, mark=plan.mark, policy=plan.policy, provenance={"m": 1}, now=NOW
    )
    assert by_parts == plan.accept("S", provenance={"m": 1}, now=NOW)
    with pytest.raises(ValueError, match="covered"):
        accept_summary("S", covered=(), mark="m0", policy=POLICY, now=NOW)
