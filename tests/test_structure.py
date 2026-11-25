"""Batches, turns and pairing (design §3.1)."""

import pytest
from builders import calls, human, results, say

from lite_context import BrokenPairing
from lite_context.structure import batches, human_turns, last_human

H1, A1 = human("h1", "q1"), calls("a1", "c1", "c2")
R1, R2 = results("r1", ("c1", "x")), results("r2", ("c2", "y"))


def test_a_batch_is_one_assistant_message_of_calls_and_every_message_holding_their_results() -> (
    None
):
    msgs = [H1, A1, R1, R2, say("s1", "done")]
    [batch] = batches(msgs)
    assert (batch.call_at, batch.result_at, batch.complete) == (1, (2, 3), True)


def test_the_last_batch_may_still_wait_for_results() -> None:
    msgs = [H1, A1, R1]
    [batch] = batches(msgs)
    assert (batch.result_at, batch.complete) == ((2,), False)


@pytest.mark.parametrize(
    "msgs",
    [
        [H1, A1, R1, say("s", "early"), R2],  # a result after the next assistant message
        [H1, A1, R1, say("s", "gave up")],  # a call never answered, and the batch is not the last
        [H1, results("r", ("zz", "?"))],  # a result without a call
        [H1, A1, R1, R2, results("r3", ("c1", "again"))],  # a second result for one call
        [H1, A1, R1, human("h2", "wait"), R2],  # a human message inside an open batch
    ],
    ids=["late-result", "unanswered", "orphan-result", "duplicate-result", "human-inside"],
)
def test_broken_pairing_is_refused(msgs: list) -> None:  # type: ignore[type-arg]
    with pytest.raises(BrokenPairing):
        batches(msgs)


def test_a_turn_starts_at_human_text_and_a_tool_result_message_does_not_start_one() -> None:
    msgs = [say("s0", "hello"), H1, A1, R1, R2, say("s1", "a"), human("h2", "q2"), say("s2", "b")]
    assert human_turns(msgs) == [(1, 6), (6, 8)]
    assert last_human(msgs) == 6


def test_no_human_text_means_no_turns() -> None:
    assert human_turns([A1, R1, R2]) == []
    assert last_human([A1, R1, R2]) is None
