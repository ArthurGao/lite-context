"""The OpenAI Chat Completions adapter round-trips losslessly (design §3.2)."""

import base64
import copy
import json
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from lite_context import Message, Opaque, Text, ToolCall, ToolResult, UnsupportedInput
from lite_context.adapters import openai

PDF_2_PAGES = base64.b64encode(
    b"%PDF-1.7\n1 0 obj << /Type /Pages >> endobj\n2 0 obj << /Type /Page >> endobj\n"
    b"3 0 obj << /Type /Page >> endobj\n"
).decode()


def roundtrip(native: list[Any]) -> list[dict[str, object]]:
    before = copy.deepcopy(native)
    converted = openai.to_messages(native)
    out = openai.from_messages(converted.messages, fixed=converted.fixed)
    assert native == before  # the input is never mutated
    return out


def call(i: str, name: str, arguments: str) -> dict[str, Any]:
    return {"id": i, "type": "function", "function": {"name": name, "arguments": arguments}}


def test_leading_system_and_developer_messages_are_fixed_and_go_back_in_front() -> None:
    native = [
        {"role": "system", "content": "be brief"},
        {"role": "developer", "content": [{"type": "text", "text": "rules"}]},
        {"role": "user", "content": "hi"},
    ]
    converted = openai.to_messages(native)
    assert converted.fixed == (native[0], native[1])
    assert converted.messages == [Message("user", (Text("hi"),), content_form="string")]
    assert converted.fixed_texts() == ("be brief", "rules")
    assert roundtrip(native) == native


def test_a_system_message_after_the_conversation_starts_is_refused() -> None:
    native = [{"role": "user", "content": "hi"}, {"role": "system", "content": "late"}]
    with pytest.raises(UnsupportedInput, match="system"):
        openai.to_messages(native)


@pytest.mark.parametrize("content", [None, "", "thinking out loud"])
def test_assistant_content_null_empty_and_text_beside_tool_calls_stay_distinct(
    content: str | None,
) -> None:
    native = [{"role": "assistant", "content": content, "tool_calls": [call("c1", "f", "{}")]}]
    parts = openai.to_messages(native).messages[0].parts
    expected_text = () if content is None else (Text(content),)
    assert parts == (
        *expected_text,
        ToolCall(id="c1", name="f", raw_args="{}", extra={"type": "function"}),
    )
    assert roundtrip(native) == native


def test_an_assistant_message_without_a_content_key_survives() -> None:
    native = [{"role": "assistant", "tool_calls": [call("c1", "f", "{}")]}]
    assert roundtrip(native) == native


def test_malformed_arguments_are_kept_verbatim() -> None:
    native = [{"role": "assistant", "content": None, "tool_calls": [call("c1", "f", '{"a": 1,')]}]
    part = openai.to_messages(native).messages[0].parts[0]
    assert isinstance(part, ToolCall)
    assert part.raw_args == '{"a": 1,'
    with pytest.raises(json.JSONDecodeError):
        part.args()
    assert roundtrip(native) == native


def test_each_tool_message_of_a_parallel_batch_stays_its_own_message() -> None:
    native = [
        {"role": "user", "content": "weather in two cities"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [call("a", "w", '{"c":"x"}'), call("b", "w", '{"c":"y"}')],
        },
        {"role": "tool", "tool_call_id": "a", "content": "sunny"},
        {"role": "tool", "tool_call_id": "b", "content": [{"type": "text", "text": "rain"}]},
        {"role": "assistant", "content": "x sunny, y rain"},
    ]
    msgs = openai.to_messages(native).messages
    assert [m.role for m in msgs] == ["user", "assistant", "user", "user", "assistant"]
    assert msgs[2].parts == (
        ToolResult(call_id="a", content=(Text("sunny"),), content_form="string"),
    )
    assert roundtrip(native) == native


def test_extra_fields_inside_function_survive() -> None:
    odd = {
        "id": "c1",
        "type": "function",
        "function": {"name": "f", "arguments": "{}", "strict": True},
    }
    native = [{"role": "assistant", "content": None, "tool_calls": [odd]}]
    assert roundtrip(native) == native


def test_name_fields_and_unknown_keys_survive() -> None:
    native = [
        {"role": "user", "name": "alice", "content": "hi"},
        {"role": "assistant", "content": "no", "refusal": None, "annotations": []},
        {"role": "assistant", "content": None, "tool_calls": [call("c1", "f", "{}")]},
        {"role": "tool", "tool_call_id": "c1", "content": "ok", "name": "f"},
    ]
    assert roundtrip(native) == native


def test_user_content_parts_become_text_and_opaque() -> None:
    image = {"type": "image_url", "image_url": {"url": "https://x/i.png", "detail": "low"}}
    pdf = {"type": "file", "file": {"file_id": "file-1"}}
    inline = {"type": "file", "file": {"file_data": "data:application/pdf;base64," + PDF_2_PAGES}}
    audio = {"type": "input_audio", "input_audio": {"data": "AAA", "format": "wav"}}
    native = [
        {"role": "user", "content": [{"type": "text", "text": "look"}, image, pdf, inline, audio]}
    ]
    parts = openai.to_messages(native).messages[0].parts
    assert parts == (
        Text("look"),
        Opaque(kind="image", raw=image),
        Opaque(kind="document", raw=pdf),
        Opaque(kind="document", raw=inline, pages=2),
        Opaque(kind="unknown", raw=audio),
    )
    assert roundtrip(native) == native


@pytest.mark.parametrize(
    "native",
    [
        [{"role": "function", "name": "f", "content": "x"}],
        [{"role": "user"}],
        [{"role": "user", "content": 3}],
        [{"role": "tool", "content": "no id"}],
        [{"role": "assistant", "content": None, "tool_calls": [{"id": "c", "type": "function"}]}],
        ["not a dict"],
    ],
)
def test_input_outside_the_supported_range_is_refused(native: list[Any]) -> None:
    with pytest.raises(UnsupportedInput):
        openai.to_messages(native)


def test_a_neutral_user_message_with_mixed_parts_keeps_their_order() -> None:
    msg = Message(
        "user",
        (
            Text("before"),
            ToolResult(call_id="a", content=(Text("1"),)),
            ToolResult(call_id="b", content=(Text("2"),)),
            Text("next?"),
        ),
    )
    assert openai.from_messages([msg]) == [
        {"role": "user", "content": "before"},
        {"role": "tool", "tool_call_id": "a", "content": "1"},
        {"role": "tool", "tool_call_id": "b", "content": "2"},
        {"role": "user", "content": "next?"},
    ]


def test_an_empty_tool_calls_list_survives() -> None:
    native = [{"role": "assistant", "content": "done", "tool_calls": []}]
    assert roundtrip(native) == native


# --- property: any conversation in the supported range round-trips ---

ids = st.text(alphabet="abcdef0123456789", min_size=1, max_size=6)
text_part = st.builds(lambda t: {"type": "text", "text": t}, st.text(max_size=20))
image_part = st.just({"type": "image_url", "image_url": {"url": "https://x/i.png"}})
file_part = st.sampled_from(
    [
        {"type": "file", "file": {"file_id": "file-1"}},
        {"type": "file", "file": {"file_data": "data:application/pdf;base64," + PDF_2_PAGES}},
    ]
)
named: st.SearchStrategy[dict[str, Any]] = st.one_of(st.just({}), st.just({"name": "alice"}))


@st.composite
def tool_call(draw: st.DrawFn) -> dict[str, Any]:
    c = call(draw(ids), draw(st.text(min_size=1, max_size=6)), draw(st.text(max_size=20)))
    if draw(st.booleans()):
        c["function"]["strict"] = True
    return c


system_message = st.builds(
    lambda r, c: {"role": r, "content": c},
    st.sampled_from(["system", "developer"]),
    st.one_of(st.text(max_size=20), st.lists(text_part, max_size=2)),
)
user_message = st.builds(
    lambda c, n: {"role": "user", "content": c, **n},
    st.one_of(
        st.text(max_size=20), st.lists(st.one_of(text_part, image_part, file_part), max_size=3)
    ),
    named,
)


@st.composite
def assistant_message(draw: st.DrawFn) -> dict[str, Any]:
    msg: dict[str, Any] = {"role": "assistant"}
    form = draw(st.sampled_from(["str", "null", "absent", "list"]))
    if form == "str":
        msg["content"] = draw(st.text(max_size=20))
    elif form == "null":
        msg["content"] = None
    elif form == "list":
        msg["content"] = draw(st.lists(text_part, max_size=2))
    calls = draw(st.sampled_from(["absent", "none", "empty", "some"]))
    if calls == "none":
        msg["tool_calls"] = None
    elif calls == "empty":
        msg["tool_calls"] = []
    elif calls == "some":
        msg["tool_calls"] = draw(st.lists(tool_call(), min_size=1, max_size=3))
    if draw(st.booleans()):
        msg["refusal"] = None
    return msg


tool_message = st.builds(
    lambda i, c, n: {"role": "tool", "tool_call_id": i, "content": c, **n},
    ids,
    st.one_of(
        st.text(max_size=20), st.lists(st.one_of(text_part, image_part, file_part), max_size=2)
    ),
    named,
)


@given(
    st.lists(system_message, max_size=2),
    st.lists(st.one_of(user_message, assistant_message(), tool_message), max_size=6),
)
def test_every_supported_conversation_round_trips(
    head: list[dict[str, Any]], body: list[dict[str, Any]]
) -> None:
    native = head + body
    assert roundtrip(native) == native


@pytest.mark.parametrize(
    "bad",
    [
        {"id": "c", "function": {"name": "f", "arguments": "{}"}},
        {"id": "c", "type": "custom", "function": {"name": "f", "arguments": "{}"}},
    ],
    ids=["no-type", "custom-type"],
)
def test_a_tool_call_that_is_not_a_function_call_is_refused(bad: dict[str, Any]) -> None:
    with pytest.raises(UnsupportedInput, match="type"):
        openai.to_messages([{"role": "assistant", "content": None, "tool_calls": [bad]}])


def test_the_sdk_shape_with_tool_calls_none_survives() -> None:
    native = [{"role": "assistant", "content": "hi", "refusal": None, "tool_calls": None}]
    assert roundtrip(native) == native


@pytest.mark.parametrize("content", [5, None, {"text": "x"}])
def test_a_system_message_whose_content_is_not_text_is_refused(content: object) -> None:
    with pytest.raises(UnsupportedInput, match="content"):
        openai.to_messages(
            [{"role": "system", "content": content}, {"role": "user", "content": "hi"}]
        )
