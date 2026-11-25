import json

import pytest

from lite_context import Message, Opaque, Text, ToolCall, ToolResult, with_keys


def test_tool_call_keeps_raw_args_verbatim_and_parses_on_demand() -> None:
    raw = '{"city": "Wellington",  "days": 3}'
    call = ToolCall(id="c1", name="weather", raw_args=raw)
    assert call.raw_args == raw
    assert call.args() == {"city": "Wellington", "days": 3}


def test_malformed_raw_args_are_kept_and_only_parsing_fails() -> None:
    call = ToolCall(id="c1", name="weather", raw_args='{"city": "Well')
    assert call.raw_args == '{"city": "Well'
    with pytest.raises(json.JSONDecodeError):
        call.args()


def test_args_that_are_not_an_object_fail_to_parse() -> None:
    with pytest.raises(ValueError, match="not a JSON object"):
        ToolCall(id="c1", name="t", raw_args="[1, 2]").args()


def test_messages_are_immutable_and_compare_by_value() -> None:
    a = Message("user", (Text("hi"),))
    b = Message("user", (Text("hi"),))
    assert a == b
    with pytest.raises(AttributeError):
        a.role = "assistant"  # type: ignore[misc]


def test_a_tool_result_holds_text_and_opaque_content() -> None:
    image = Opaque(kind="image", raw={"type": "image"})
    result = ToolResult(call_id="c1", content=(Text("chart below"), image), is_error=False)
    assert result.content == (Text("chart below"), image)


def test_with_keys_sets_one_key_per_message_and_changes_nothing_else() -> None:
    msgs = [Message("user", (Text("a"),)), Message("assistant", (Text("b"),))]
    keyed = with_keys(msgs, ["m1", "m2"])
    assert [m.key for m in keyed] == ["m1", "m2"]
    assert [m.parts for m in keyed] == [m.parts for m in msgs]
    assert [m.key for m in msgs] == [None, None]  # the input is untouched


def test_with_keys_refuses_a_count_mismatch_and_duplicate_keys() -> None:
    msgs = [Message("user", (Text("a"),)), Message("assistant", (Text("b"),))]
    with pytest.raises(ValueError, match="2 messages but 1 keys"):
        with_keys(msgs, ["m1"])
    with pytest.raises(ValueError, match="duplicate key 'm1'"):
        with_keys(msgs, ["m1", "m1"])


def test_an_unknown_role_is_refused() -> None:
    with pytest.raises(ValueError, match="role"):
        Message("system", (Text("x"),))  # type: ignore[arg-type]
