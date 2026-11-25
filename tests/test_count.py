import pytest

from lite_context import (
    Budget,
    CharEstimate,
    DoesNotFit,
    Message,
    Opaque,
    Text,
    ToolCall,
    ToolResult,
    count_message,
    count_messages,
)

EST = CharEstimate()  # 3.5 chars per token, 4 tokens per message


def test_text_is_characters_over_three_and_a_half_rounded_up() -> None:
    assert EST.text("") == 0
    assert EST.text("abc") == 1
    assert EST.text("a" * 7) == 2
    assert EST.text("a" * 8) == 3


def test_a_message_costs_its_parts_plus_the_framing_overhead() -> None:
    msg = Message(
        "assistant",
        (Text("a" * 7), ToolCall(id="c1", name="abcdefg", raw_args='{"q":"abcdefg"}')),
    )
    # 4 framing + 2 text + 2 name + ceil(15 / 3.5) = 5 args
    assert count_message(msg, EST) == 13


def test_a_tool_result_costs_its_content() -> None:
    msg = Message("user", (ToolResult(call_id="c1", content=(Text("a" * 35),)),))
    assert count_message(msg, EST) == 4 + 10


def test_count_messages_sums_messages() -> None:
    msgs = [Message("user", (Text("a" * 7),)), Message("assistant", (Text("a" * 7),))]
    assert count_messages(msgs, EST) == 12


def test_an_opaque_part_with_known_tokens_uses_them_and_the_counter_is_not_asked() -> None:
    class NoOpaque(CharEstimate):
        def opaque(self, part: Opaque) -> int:
            raise AssertionError("the counter was asked for a cost the caller gave")

    msg = Message("user", (Opaque(kind="image", raw={"type": "image"}, tokens=900),))
    assert count_message(msg, NoOpaque()) == 4 + 900


def test_an_image_without_tokens_costs_the_image_default() -> None:
    assert EST.opaque(Opaque(kind="image", raw={"type": "image"})) == 1600


def test_an_opaque_part_with_its_own_text_costs_that_text() -> None:
    assert EST.opaque(Opaque(kind="thinking", raw=None, text="a" * 35)) == 10


def test_a_pdf_costs_per_page() -> None:
    assert EST.opaque(Opaque(kind="document", raw=None, pages=3)) == 3 * 2000


def test_an_opaque_part_without_measures_costs_the_default_not_its_raw_size() -> None:
    raw = {"type": "server_tool_use", "payload": "x" * 100_000}
    assert EST.opaque(Opaque(kind="unknown", raw=raw)) == 1500


def test_budget_available_subtracts_output_and_fixed_text() -> None:
    budget = Budget(limit=1000, reserve_output=200)
    assert budget.available(["a" * 35, 50], EST) == 1000 - 200 - 10 - 50


def test_fixed_content_over_the_budget_does_not_fit_before_any_cutting() -> None:
    budget = Budget(limit=100, reserve_output=50)
    with pytest.raises(DoesNotFit) as caught:
        budget.available(["a" * 350], EST)
    report = caught.value.report
    assert (report.limit, report.reserve_output, report.fixed_tokens) == (100, 50, 100)


def test_a_budget_must_leave_room_for_input() -> None:
    with pytest.raises(ValueError, match="reserve_output"):
        Budget(limit=100, reserve_output=100)


@pytest.mark.parametrize(
    "kwargs",
    [{"chars_per_token": 0}, {"chars_per_token": -1}, {"per_message": -1}, {"image_tokens": -1}],
)
def test_a_char_estimate_with_impossible_numbers_is_refused(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        CharEstimate(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("fixed", [[-5], [True]], ids=["negative", "bool"])
def test_fixed_token_counts_must_be_non_negative_ints(fixed: list[int]) -> None:
    with pytest.raises(ValueError, match="fixed"):
        Budget(limit=10, reserve_output=0).available(fixed, EST)
