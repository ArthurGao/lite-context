"""`fit()`: the flow, the water marks, state and replay, and the rules of design §5.2."""

import copy
from collections.abc import Sequence
from typing import Literal

import pytest
from builders import COUNTER, calls, human, image, results, say, thinking_call

from lite_context import (
    BrokenPairing,
    Budget,
    ContextState,
    DoesNotFit,
    Message,
    Opaque,
    ProtectedChange,
    StaleState,
    Text,
    ToolCall,
    ToolResult,
    UnreportedChange,
    fit,
)
from lite_context.steps import (
    DROPPED,
    MASKED_OPAQUE,
    MASKED_RESULT,
    TRUNCATED,
    Action,
    DropOldest,
    Entry,
    MaskOpaque,
    MaskToolResults,
    Step,
    StepContext,
    TruncateToolResults,
    Unit,
)

MASK = MASKED_RESULT.format(name="search")


def conversation(turns: int, size: int = 200) -> list[Message]:
    msgs: list[Message] = []
    for t in range(1, turns + 1):
        msgs += [
            human(f"h{t}", "q"),
            calls(f"a{t}", f"c{t}"),
            results(f"r{t}", (f"c{t}", "x" * size)),
            say(f"s{t}", "ok"),
        ]
    return msgs


def result_text(m: Message) -> str:
    part = m.parts[0]
    assert isinstance(part, ToolResult)
    text = part.content[0]
    assert isinstance(text, Text)
    return text.text


def by_key(msgs: Sequence[Message]) -> dict[str, Message]:
    return {m.key: m for m in msgs if m.key}


# --- the flow ---


def test_a_history_that_fits_comes_back_unchanged_with_its_numbers() -> None:
    msgs = conversation(2)  # 2 * (1 + 8 + 200 + 2) = 422
    fitted = fit(msgs, budget=Budget(1000, 100), fixed=["s" * 50], counter=COUNTER)
    assert fitted.messages == msgs
    r = fitted.report
    assert (r.available, r.fixed_tokens, r.tokens_before, r.tokens_after) == (850, 50, 422, 422)
    assert (r.actions, r.compaction_due) == ((), False)
    assert fitted.state == ContextState()


def test_the_input_is_untouched_and_the_same_input_gives_the_same_output() -> None:
    msgs = conversation(3)
    before = copy.deepcopy(msgs)
    steps: list[Step] = [TruncateToolResults(max_tokens=50), MaskToolResults(keep_last=1)]
    one = fit(msgs, budget=Budget(300, 0), steps=steps, counter=COUNTER)
    two = fit(msgs, budget=Budget(300, 0), steps=steps, counter=COUNTER)
    assert msgs == before
    assert (one.messages, one.state, one.report) == (two.messages, two.state, two.report)


def test_cutting_steps_stop_at_the_low_water_mark() -> None:
    msgs = conversation(3)  # 3 * 211 = 633
    # available 650: high 585, target 455. Masking saves 200 - len(MASK) per batch.
    fitted = fit(msgs, budget=Budget(650, 0), steps=[MaskToolResults(keep_last=0)], counter=COUNTER)
    out = by_key(fitted.messages)
    saved = 200 - len(MASK)
    assert [result_text(out[k]) for k in ("r1", "r2", "r3")] == [MASK, MASK, "x" * 200]
    assert fitted.report.tokens_after == 633 - 2 * saved <= 455
    assert [(a.step, a.ref, a.what, a.tokens_saved) for a in fitted.report.actions] == [
        ("mask_tool_results", "r1", "masked", saved),
        ("mask_tool_results", "r2", "masked", saved),
    ]
    assert fitted.state.cuts["mask_tool_results"].through == "r2"


def test_nothing_is_cut_below_the_high_water_mark() -> None:
    msgs = conversation(3)  # 633 <= 0.9 * 720 = 648
    fitted = fit(msgs, budget=Budget(720, 0), steps=[MaskToolResults(keep_last=0)], counter=COUNTER)
    assert fitted.messages == msgs


def test_when_every_step_has_run_and_it_still_does_not_fit_it_raises_with_the_report() -> None:
    msgs = conversation(3)
    with pytest.raises(DoesNotFit) as caught:
        fit(msgs, budget=Budget(300, 0), steps=[MaskToolResults(keep_last=1)], counter=COUNTER)
    report = caught.value.report
    assert [a.ref for a in report.actions] == ["r1", "r2"]
    assert report.tokens_after == 633 - 2 * (200 - len(MASK)) > report.available == 300


def test_compaction_is_due_when_the_cut_history_is_still_over_the_high_water_mark() -> None:
    msgs = conversation(3)
    fitted = fit(msgs, budget=Budget(400, 0), steps=[MaskToolResults(keep_last=1)], counter=COUNTER)
    assert fitted.report.tokens_after == 633 - 2 * (200 - len(MASK))  # 369: fits 400, over 360
    assert fitted.report.compaction_due is True


def test_high_and_low_water_marks_are_checked() -> None:
    with pytest.raises(ValueError, match="water"):
        fit([], budget=Budget(100, 0), high=0.5, low=0.7, counter=COUNTER)


def test_two_steps_with_one_name_are_refused() -> None:
    with pytest.raises(ValueError, match="mask_tool_results"):
        fit([], budget=Budget(100, 0), steps=[MaskToolResults(), MaskToolResults()])


# --- the steps ---


def test_truncation_always_runs_and_keeps_the_head_with_a_fixed_marker() -> None:
    msgs = [human("h", "q"), calls("a", "c"), results("r", ("c", "abcdefghij" * 20))]
    fitted = fit(msgs, budget=Budget(10_000, 0), steps=[TruncateToolResults(20)], counter=COUNTER)
    marker = TRUNCATED.format(n=180)
    assert result_text(fitted.messages[2]) == "abcdefghij" * 2 + "\n" + marker
    assert [(a.ref, a.what) for a in fitted.report.actions] == [("r", "truncated")]


def test_head_tail_truncation_keeps_both_ends() -> None:
    msgs = [human("h", "q"), calls("a", "c"), results("r", ("c", "H" * 10 + "m" * 80 + "T" * 10))]
    step = TruncateToolResults(20, keep="head_tail")
    fitted = fit(msgs, budget=Budget(10_000, 0), steps=[step], counter=COUNTER)
    assert (
        result_text(fitted.messages[2])
        == "H" * 10 + "\n" + TRUNCATED.format(n=80) + "\n" + "T" * 10
    )


def test_the_newest_tool_result_is_truncated_rather_than_failing_every_call() -> None:
    msgs = [human("h", "q"), calls("a", "c"), results("r", ("c", "x" * 5000))]
    fitted = fit(msgs, budget=Budget(500, 0), steps=[TruncateToolResults(100)], counter=COUNTER)
    assert result_text(fitted.messages[2]).startswith("x" * 100 + "\n[cut: 4900 ")


def test_mask_keeps_the_last_n_batches_counting_a_parallel_batch_once() -> None:
    msgs = [
        human("h1", "q"),
        calls("a1", "c1"),
        results("r1", ("c1", "x" * 200)),
        human("h2", "q"),
        calls("a2", "c2", "c3"),
        results("r2", ("c2", "y" * 200), ("c3", "z" * 200)),
    ]
    fitted = fit(msgs, budget=Budget(600, 0), steps=[MaskToolResults(keep_last=1)], counter=COUNTER)
    out = by_key(fitted.messages)
    assert result_text(out["r1"]) == MASK
    assert out["r2"] == msgs[5]  # one batch of two calls is the one kept


def test_a_pinned_batch_is_skipped_and_the_next_one_masked() -> None:
    msgs = conversation(3)
    msgs[2] = results("r1", ("c1", "x" * 200), pinned=True)
    fitted = fit(msgs, budget=Budget(650, 0), steps=[MaskToolResults(keep_last=0)], counter=COUNTER)
    out = by_key(fitted.messages)
    assert [result_text(out[k]) for k in ("r1", "r2", "r3")] == ["x" * 200, MASK, MASK]


def test_mask_opaque_removes_old_images_but_not_the_last_human_turns_or_thinking() -> None:
    msgs = [
        image("i1"),
        thinking_call("a1", "c1", "t" * 300),
        results("r1", ("c1", "done")),
        image("i2"),
    ]
    fitted = fit(msgs, budget=Budget(500, 0), steps=[MaskOpaque(keep_last=0)], counter=COUNTER)
    out = by_key(fitted.messages)
    assert out["i1"].parts == (Text("look"), Text(MASKED_OPAQUE.format(kind="image")))
    assert out["i2"] == msgs[3]  # the last human turn
    assert out["a1"] == msgs[1]  # thinking is never touched


def test_drop_oldest_drops_whole_turns_and_says_so_up_front() -> None:
    msgs = conversation(4)  # 844
    step = DropOldest(keep_last_turns=2)
    fitted = fit(msgs, budget=Budget(600, 0), steps=[step], counter=COUNTER)
    placeholder = DROPPED.format(n=8)
    assert fitted.messages[0] == Message("user", (Text(placeholder),), key=None)
    assert [m.key for m in fitted.messages[1:]] == ["h3", "a3", "r3", "s3", "h4", "a4", "r4", "s4"]
    first = fitted.report.actions[0]
    assert (first.ref, first.what, first.tokens_saved) == (
        "+drop_oldest",
        "inserted",
        -len(placeholder),
    )


def test_drop_oldest_never_drops_a_turn_holding_a_pinned_message() -> None:
    msgs = conversation(3)
    msgs[0] = human("h1", "q", pinned=True)
    fitted = fit(
        msgs, budget=Budget(500, 0), steps=[DropOldest(keep_last_turns=1)], counter=COUNTER
    )
    assert [m.key for m in fitted.messages if m.key] == [
        "h1",
        "a1",
        "r1",
        "s1",
        "h3",
        "a3",
        "r3",
        "s3",
    ]


# --- state: replay, advance only when needed, stale ---


def test_with_state_old_messages_change_only_on_the_turns_that_cross_the_high_water_mark() -> None:
    history: list[Message] = []
    state = ContextState()
    previous: list[Message] = []
    advanced = 0
    for t in range(1, 21):
        history += [
            human(f"h{t}", "q"),
            calls(f"a{t}", f"c{t}"),
            results(f"r{t}", (f"c{t}", "x" * 200)),
            say(f"s{t}", "ok"),
        ]
        fitted = fit(
            history,
            budget=Budget(3000, 0),
            steps=[MaskToolResults(keep_last=2)],
            state=state,
            counter=COUNTER,
        )
        moved = fitted.state != state
        if not moved:  # nothing new was cut: the previous request is a prefix of this one
            assert fitted.messages[: len(previous)] == previous
        advanced += moved
        state, previous = fitted.state, fitted.messages
    assert 1 <= advanced <= 4


def test_state_is_replayed_even_when_the_history_would_fit_without_it() -> None:
    msgs = conversation(3)
    first = fit(msgs, budget=Budget(650, 0), steps=[MaskToolResults(keep_last=0)], counter=COUNTER)
    again = fit(
        msgs,
        budget=Budget(10_000, 0),
        steps=[MaskToolResults(keep_last=0)],
        state=first.state,
        counter=COUNTER,
    )
    assert again.messages == first.messages


def test_state_from_a_history_that_changed_before_the_cut_is_stale() -> None:
    msgs = conversation(3)
    first = fit(msgs, budget=Budget(650, 0), steps=[MaskToolResults(keep_last=0)], counter=COUNTER)
    with pytest.raises(StaleState):
        fit(
            msgs[4:],
            budget=Budget(650, 0),
            steps=[MaskToolResults(keep_last=0)],
            state=first.state,
            counter=COUNTER,
        )


def test_without_keys_no_cuts_are_kept() -> None:
    msgs = [Message(m.role, m.parts) for m in conversation(3)]
    fitted = fit(msgs, budget=Budget(650, 0), steps=[MaskToolResults(keep_last=0)], counter=COUNTER)
    assert fitted.state == ContextState()
    assert [a.ref for a in fitted.report.actions] == ["#2", "#6"]


def test_state_survives_json_and_an_unknown_version_is_refused() -> None:
    msgs = conversation(3)
    state = fit(
        msgs, budget=Budget(650, 0), steps=[MaskToolResults(keep_last=0)], counter=COUNTER
    ).state
    assert ContextState.from_json(state.to_json()) == state
    with pytest.raises(ValueError, match="version"):
        ContextState.from_json('{"version": 99, "cuts": {}}')


# --- the run-time checks on every step, the host's own included ---


class Rogue:
    """A host's cutting step that does what it is told by `harm`."""

    name = "rogue"
    always: Literal[False] = False

    def __init__(self, harm: str) -> None:
        self.harm = harm

    def units(self, entries: Sequence[Entry], ctx: StepContext) -> list[Unit]:
        return [Unit(refs=(entries[0].ref,))]

    def apply(
        self, entries: Sequence[Entry], units: Sequence[Unit], ctx: StepContext
    ) -> tuple[list[Entry], list[Action]]:
        out = list(entries)
        if self.harm == "silent-edit":
            out[2] = Entry(out[2].ref, results("r1", ("c1", "edited")))
            return out, []
        if self.harm == "break-pair":
            ref = out[2].ref
            del out[2]
            return out, [Action(self.name, ref, "dropped", 0)]
        if self.harm == "duplicate":
            out.append(Entry(out[2].ref, out[2].message))
            return out, [Action(self.name, out[2].ref, "copied", 0)]
        if self.harm == "touch-pinned":
            out[0] = Entry(out[0].ref, human("h1", "changed", pinned=True))
            return out, [Action(self.name, out[0].ref, "edited", 0)]
        raise AssertionError(self.harm)


@pytest.mark.parametrize(
    ("harm", "error"),
    [
        ("silent-edit", UnreportedChange),
        ("break-pair", BrokenPairing),
        ("duplicate", UnreportedChange),
        ("touch-pinned", ProtectedChange),
    ],
)
def test_a_step_that_breaks_a_rule_is_caught_at_run_time(harm: str, error: type[Exception]) -> None:
    msgs = conversation(3)
    msgs[0] = human("h1", "q", pinned=True)
    with pytest.raises(error):
        fit(msgs, budget=Budget(500, 0), steps=[Rogue(harm)], counter=COUNTER)


def test_an_unfinished_batch_is_never_masked() -> None:
    msgs = [
        *conversation(2),
        human("h3", "q"),
        calls("a3", "c3", "c4"),
        results("r3", ("c3", "x" * 400)),
    ]
    fitted = fit(msgs, budget=Budget(800, 0), steps=[MaskToolResults(keep_last=0)], counter=COUNTER)
    assert by_key(fitted.messages)["r3"] == msgs[-1]
    assert [a.ref for a in fitted.report.actions] == ["r1", "r2"]


class ThinkingThief:
    name = "thief"
    always: Literal[True] = True

    def apply(self, entries: Sequence[Entry], ctx: StepContext) -> tuple[list[Entry], list[Action]]:
        out = list(entries)
        m = out[1].message
        out[1] = Entry(out[1].ref, Message(m.role, m.parts[1:], key=m.key))
        return out, [Action(self.name, out[1].ref, "edited", 0)]


def test_a_step_that_removes_a_thinking_block_is_caught() -> None:
    msgs = [
        human("h1", "q"),
        thinking_call("a1", "c1", "hm"),
        results("r1", ("c1", "x")),
        human("h2", "q"),
    ]
    with pytest.raises(ProtectedChange, match="thinking"):
        fit(msgs, budget=Budget(10_000, 0), steps=[ThinkingThief()], counter=COUNTER)


# --- regressions from the C1a review ---


def _screenshot_turn(t: int) -> list[Message]:
    shot = Opaque(kind="image", raw={"i": t})
    return [
        human(f"h{t}", f"q{t}"),
        calls(f"a{t}", f"c{t}", name="shot"),
        Message("user", (ToolResult(f"c{t}", (Text("r" * 50), shot)),), key=f"r{t}"),
        say(f"s{t}", "ok"),
    ]


def test_two_cutting_steps_keep_working_as_the_history_grows() -> None:
    steps: list[Step] = [MaskToolResults(keep_last=1), MaskOpaque(keep_last=0)]
    history: list[Message] = []
    state = ContextState()
    previous: list[Message] = []
    overtaken = False  # the case that went stale: mask_tool_results moves past mask_opaque's cut
    for t in range(1, 16):  # turn 15 overtakes; turn 16 no longer fits 1200
        history += _screenshot_turn(t)
        fitted = fit(history, budget=Budget(1200, 0), steps=steps, state=state, counter=COUNTER)
        if fitted.state == state:
            assert fitted.messages[: len(previous)] == previous
        cuts = fitted.state.cuts
        if "mask_opaque" in state.cuts and "mask_tool_results" in cuts:
            order = [m.key for m in history]
            overtaken |= order.index(cuts["mask_tool_results"].through) >= order.index(
                state.cuts["mask_opaque"].through
            )
        state, previous = fitted.state, fitted.messages
    assert overtaken


def test_protection_moving_off_an_old_image_does_not_make_the_state_stale() -> None:
    def shot(k: str) -> Opaque:
        return Opaque(kind="image", raw={"i": k})

    msgs = [
        Message("user", (Text("q"), shot("A")), key="h1"),
        Message(
            "assistant",
            (ToolCall("c1", "s", "{}"), ToolCall("c2", "s", "{}")),
            key="a1",
        ),
        Message("user", (ToolResult("c1", (Text("x" * 10), shot("B"))),), key="r1"),
        Message("user", (ToolResult("c2", (shot("C"),)),), key="r2"),
        say("s1", "ok"),
    ]
    step = MaskOpaque(keep_last=1)
    first = fit(msgs, budget=Budget(330, 0), steps=[step], counter=COUNTER)
    assert [a.ref for a in first.report.actions] == ["r1"]
    grown = [*msgs, human("h2", "next")]
    second = fit(grown, budget=Budget(330, 0), steps=[step], state=first.state, counter=COUNTER)
    assert by_key(second.messages)["r1"] == by_key(first.messages)["r1"]


@pytest.mark.parametrize("key", ["#1", "+drop_oldest"])
def test_keys_that_look_like_internal_refs_are_refused(key: str) -> None:
    msgs = [human(key, "x" * 50), Message("assistant", (Text("y" * 50),))]
    with pytest.raises(ValueError, match="reserved"):
        fit(msgs, budget=Budget(60, 0), counter=COUNTER)


def test_a_cut_that_would_cost_more_than_it_saves_is_not_taken() -> None:
    msgs = [human("h1", "a"), say("s1", "b"), human("h2", "c"), say("s2", "d")] * 1
    msgs += [human("h3", "e" * 20)]
    fitted = fit(msgs, budget=Budget(26, 0), steps=[DropOldest(keep_last_turns=1)], counter=COUNTER)
    assert fitted.messages == msgs
    assert fitted.report.actions == ()


def test_truncation_limits_a_whole_result_not_each_text_part() -> None:
    parts = tuple(Text("abcde" * 4) for _ in range(5))  # five parts of 20, 100 in all
    msgs = [human("h", "q"), calls("a", "c"), Message("user", (ToolResult("c", parts),), key="r")]
    fitted = fit(msgs, budget=Budget(10_000, 0), steps=[TruncateToolResults(30)], counter=COUNTER)
    result = fitted.messages[2].parts[0]
    assert isinstance(result, ToolResult)
    joined = "\n".join(p.text for p in parts)
    assert result.content == (Text(joined[:30] + "\n" + TRUNCATED.format(n=len(joined) - 30)),)


def test_head_and_tail_never_overlap_even_with_a_counter_that_is_not_additive() -> None:
    from lite_context import CharEstimate

    class Lumpy(CharEstimate):
        def text(self, s: str) -> int:  # a short piece costs nothing, like merged BPE tokens
            return 0 if len(s) <= 30 else len(s)

    lumpy = Lumpy(per_message=0)
    msgs = [human("h", "q"), calls("a", "c"), results("r", ("c", "H" * 20 + "T" * 20))]
    step = TruncateToolResults(10, keep="head_tail")
    fitted = fit(msgs, budget=Budget(10_000, 0), steps=[step], counter=lumpy)
    text = result_text(fitted.messages[2])
    assert "-" not in text.split("[cut: ")[-1].split(" ")[0]
    assert text.count("H") + text.count("T") <= 40


@pytest.mark.parametrize(
    "text",
    ['{"version": 1, "cuts": []}', '{"version": 1, "cuts": {"m": {"count": 1}}}', "[1]", "nope"],
)
def test_a_malformed_state_is_a_value_error(text: str) -> None:
    with pytest.raises(ValueError):
        ContextState.from_json(text)


# --- regressions from the second C1a review ---


def test_images_in_one_message_are_cut_together_so_replay_cuts_no_more() -> None:
    def shot(k: str) -> Opaque:
        return Opaque(kind="image", raw={"i": k})

    def two_shots(t: int) -> list[Message]:
        return [
            human(f"h{t}", "q"),
            calls(f"a{t}", f"c{t}"),
            Message("user", (ToolResult(f"c{t}", (shot(f"{t}a"), shot(f"{t}b"))),), key=f"r{t}"),
            say(f"s{t}", "ok"),
        ]

    history: list[Message] = []
    state = ContextState()
    previous: list[Message] = []
    for t in range(1, 8):
        history += two_shots(t)
        fitted = fit(
            history,
            budget=Budget(800, 0),
            steps=[MaskOpaque(keep_last=1)],
            state=state,
            counter=COUNTER,
        )
        for m in fitted.messages:  # never one image masked and its sibling kept
            if m.key and m.key.startswith("r"):
                kinds = [type(c).__name__ for c in m.parts[0].content]  # type: ignore[union-attr]
                assert len(set(kinds)) == 1, (t, m.key, kinds)
        if fitted.state == state:
            assert fitted.messages[: len(previous)] == previous
        state, previous = fitted.state, fitted.messages


def test_a_cut_is_found_even_when_masking_some_results_makes_them_longer() -> None:
    msgs: list[Message] = []
    for t in range(12):
        size = 400 if t == 0 else 1
        msgs += [
            human(f"h{t}", "q"),
            calls(f"a{t}", f"c{t}"),
            results(f"r{t}", (f"c{t}", "x" * size)),
        ]
    fitted = fit(msgs, budget=Budget(344, 0), steps=[MaskToolResults(keep_last=1)], counter=COUNTER)
    assert [a.ref for a in fitted.report.actions] == ["r0"]


def test_truncation_never_makes_a_result_longer() -> None:
    msgs = [human("h", "q"), calls("a", "c"), results("r", ("c", "x" * 12))]
    fitted = fit(msgs, budget=Budget(10_000, 0), steps=[TruncateToolResults(10)], counter=COUNTER)
    assert fitted.messages == msgs


def test_truncation_handles_the_same_text_object_twice() -> None:
    t = Text("abcdefghij" * 5)
    img = Opaque(kind="image", raw={"i": 1})
    msgs = [
        human("h", "q"),
        calls("a", "c"),
        Message("user", (ToolResult("c", (t, img, t)),), key="r"),
    ]
    fitted = fit(msgs, budget=Budget(10_000, 0), steps=[TruncateToolResults(40)], counter=COUNTER)
    content = fitted.messages[2].parts[0].content  # type: ignore[union-attr]
    assert [type(c).__name__ for c in content] == ["Text", "Opaque"]


def test_when_no_cut_reaches_the_low_mark_the_cheapest_one_that_fits_is_taken() -> None:
    msgs: list[Message] = []
    for t in range(12):
        size = 400 if t == 0 else 1
        msgs += [
            human(f"h{t}", "q"),
            calls(f"a{t}", f"c{t}"),
            results(f"r{t}", (f"c{t}", "x" * size)),
        ]
    # 519 in all; available 200: masking r0 gives 185 (over the low mark, 140, but fits)
    fitted = fit(msgs, budget=Budget(200, 0), steps=[MaskToolResults(keep_last=1)], counter=COUNTER)
    assert [a.ref for a in fitted.report.actions] == ["r0"]


class PerImage:
    """A host's step with one unit per image, several in one message."""

    name = "per_image"
    always: Literal[False] = False

    def units(self, entries: Sequence[Entry], ctx: StepContext) -> list[Unit]:
        return [
            Unit((e.ref,), (i, j))
            for i, e in enumerate(entries)
            if e.ref not in ctx.protected
            for j, p in enumerate(e.message.parts)
            if isinstance(p, Opaque) and p.kind == "image"
        ]

    def apply(
        self, entries: Sequence[Entry], units: Sequence[Unit], ctx: StepContext
    ) -> tuple[list[Entry], list[Action]]:
        spots = {(u.at[0], u.at[1]) for u in units}
        out, actions = list(entries), []
        for i, e in enumerate(entries):
            parts = tuple(
                Text("[gone]") if (i, j) in spots else p for j, p in enumerate(e.message.parts)
            )
            if parts != e.message.parts:
                out[i] = Entry(e.ref, Message(e.message.role, parts, key=e.message.key))
                actions.append(Action(self.name, e.ref, "masked", 0))
        return out, actions


def test_a_host_steps_units_in_one_message_are_cut_together() -> None:
    def gallery(t: int) -> Message:
        shots = tuple(Opaque(kind="image", raw={"t": t, "j": j}) for j in range(3))
        return Message("user", (Text("see"), *shots), key=f"g{t}")

    history: list[Message] = []
    state = ContextState()
    previous: list[Message] = []
    for t in range(1, 8):
        history += [gallery(t), say(f"s{t}", "ok")]
        fitted = fit(
            history, budget=Budget(1500, 0), steps=[PerImage()], state=state, counter=COUNTER
        )
        for m in fitted.messages:
            kinds = {type(p).__name__ for p in m.parts[1:]}
            assert len(kinds) <= 1, (t, m.key, kinds)
        if fitted.state == state:
            assert fitted.messages[: len(previous)] == previous
        state, previous = fitted.state, fitted.messages
