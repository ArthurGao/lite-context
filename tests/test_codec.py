"""The JSON codec for neutral messages (design §8): what a store writes and reads back."""

import json
from typing import Any

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from test_adapter_anthropic import assistant_message, user_message

from lite_context import Message, Opaque, Text, ToolCall, ToolResult
from lite_context.adapters import anthropic, openai
from lite_context.codec import _storable, message_from_dict, message_to_dict
from lite_context.testing import sample_conversations


def storable(value: object) -> bool:
    try:
        _storable(value)
    except ValueError:
        return False
    return True


def roundtrip(m: Message) -> Message:
    # Through real JSON text, as a store keeps it.
    return message_from_dict(json.loads(json.dumps(message_to_dict(m))))


def test_every_sample_conversation_round_trips() -> None:
    for conversation in sample_conversations():
        assert [roundtrip(m) for m in conversation] == conversation


@given(st.lists(st.one_of(user_message, assistant_message), max_size=6))
def test_every_message_an_adapter_produces_round_trips(native: list[dict[str, Any]]) -> None:
    # NUL and lone surrogates are refused on purpose (the next tests); every other text survives.
    assume(storable(native))
    for m in anthropic.to_messages(native):
        assert roundtrip(m) == m


def test_every_field_of_every_part_is_kept() -> None:
    m = Message(
        "user",
        (
            Text("hi", extra={"cache_control": {"type": "ephemeral"}}),
            ToolResult(
                "c1",
                (Text("r"), Opaque("image", {"src": 1}, tokens=7)),
                is_error=False,
                content_form="list",
                extra={"x": [1, 2]},
            ),
            Opaque("document", {"d": "x"}, text="t", pages=3),
        ),
        key="k1",
        pinned=True,
        content_form="list",
        extra={"name": "n"},
    )
    assert roundtrip(m) == m
    call = Message("assistant", (ToolCall("c1", "search", '{"q": 1', extra={"i": 0}),))
    assert roundtrip(call) == call  # malformed arguments stay verbatim


def test_an_openai_conversation_round_trips_too() -> None:
    native = [
        {"role": "user", "content": "find it"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "s", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "found"},
    ]
    for m in openai.to_messages(native).messages:
        assert roundtrip(m) == m


@pytest.mark.parametrize(
    "raw", [{"t": (1, 2)}, {1: "int key"}, object(), {"f": float("nan")}], ids=str
)
def test_content_json_cannot_give_back_equal_is_refused(raw: object) -> None:
    m = Message("user", (Opaque("image", raw),))
    with pytest.raises(ValueError, match="JSON"):
        message_to_dict(m)


def test_an_unknown_version_or_a_malformed_record_is_refused() -> None:
    good = message_to_dict(Message("user", (Text("x"),)))
    with pytest.raises(ValueError, match="version"):
        message_from_dict({**good, "v": 99})
    with pytest.raises(ValueError, match="not a message"):
        message_from_dict({**good, "parts": [{"type": "sound"}]})
    with pytest.raises(ValueError, match="not a message"):
        message_from_dict({**good, "role": "system"})


@pytest.mark.parametrize(
    "bad",
    [
        Message("user", (Text("x"),), pinned=1),  # type: ignore[arg-type]
        Message("user", (Opaque("image", None, tokens=True),)),
        Message("user", (ToolResult("c", (ToolResult("d", ()),)),)),  # type: ignore[arg-type]
        Message("user", (ToolResult("c", (), is_error=0),)),  # type: ignore[arg-type]
        Message("user", (Text(7),)),  # type: ignore[arg-type]
        Message("user", (Opaque("sound", {}),)),  # type: ignore[arg-type]
    ],
    ids=["pinned-int", "tokens-bool", "nested-result", "is_error-int", "text-int", "kind"],
)
def test_what_could_not_be_read_back_is_refused_when_written(bad: Message) -> None:
    # A transcript is append-only: a record that cannot be read would make it unreadable for good.
    with pytest.raises(ValueError, match="not a message"):
        message_to_dict(bad)


@pytest.mark.parametrize(
    "bad",
    [
        Message("user", (Text("a\x00b"),)),
        Message("user", (Text("a\ud800b"),)),
        Message("user", (Opaque("image", {"k\x00": "v"}),)),
        Message("assistant", (ToolCall("c", "s", '{"q": "\x00"}'),)),
    ],
    ids=["nul", "lone-surrogate", "nul-in-a-key", "nul-in-arguments"],
)
def test_characters_postgresql_cannot_store_are_refused_by_every_store(bad: Message) -> None:
    with pytest.raises(ValueError, match=r"NUL|surrogate"):
        message_to_dict(bad)
