"""The Anthropic Messages adapter round-trips losslessly (design §3.2)."""

import base64
import copy
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from lite_context import Message, Opaque, Text, ToolCall, ToolResult, UnsupportedInput
from lite_context.adapters import anthropic

CACHE = {"type": "ephemeral"}


def roundtrip(native: list[Any]) -> list[dict[str, object]]:
    before = copy.deepcopy(native)
    out = anthropic.from_messages(anthropic.to_messages(native))
    assert native == before  # the input is never mutated
    return out


def test_string_content_and_a_single_text_block_stay_distinct() -> None:
    native = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": [{"type": "text", "text": "hello"}]},
    ]
    msgs = anthropic.to_messages(native)
    assert msgs[0].parts == (Text("hello"),)
    assert msgs[1].parts == (Text("hello"),)
    assert roundtrip(native) == native


def test_cache_control_on_blocks_and_extra_message_fields_survive() -> None:
    native = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "big doc", "cache_control": CACHE}],
            "x-trace": "t1",
        },
    ]
    msgs = anthropic.to_messages(native)
    assert msgs[0].parts == (Text("big doc", extra={"cache_control": CACHE}),)
    assert roundtrip(native) == native


def test_tool_use_input_becomes_raw_args_and_comes_back_equal() -> None:
    native = [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "checking"},
                {"type": "tool_use", "id": "tu1", "name": "search", "input": {"q": "千里", "n": 3}},
                {
                    "type": "tool_use",
                    "id": "tu2",
                    "name": "noop",
                    "input": {},
                    "cache_control": CACHE,
                },
            ],
        },
    ]
    call = anthropic.to_messages(native)[0].parts[1]
    assert isinstance(call, ToolCall)
    assert (call.id, call.name, call.args()) == ("tu1", "search", {"q": "千里", "n": 3})
    assert roundtrip(native) == native


def test_tool_result_content_forms_and_is_error_presence_survive() -> None:
    image = {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "iVBO"},
    }
    native = [
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "a", "content": "plain"},
                {
                    "type": "tool_result",
                    "tool_use_id": "b",
                    "content": [{"type": "text", "text": "chart"}, image],
                },
                {"type": "tool_result", "tool_use_id": "c"},
                {"type": "tool_result", "tool_use_id": "d", "content": "boom", "is_error": True},
                {"type": "tool_result", "tool_use_id": "e", "content": "fine", "is_error": False},
                {"type": "tool_result", "tool_use_id": "f", "content": "x", "cache_control": CACHE},
            ],
        },
    ]
    parts = anthropic.to_messages(native)[0].parts
    assert all(isinstance(p, ToolResult) for p in parts)
    b = parts[1]
    assert isinstance(b, ToolResult)
    assert b.content == (Text("chart"), Opaque(kind="image", raw=image))
    results = [p for p in parts if isinstance(p, ToolResult)]
    assert [p.is_error for p in results] == [None, None, None, True, False, None]
    assert [p.content_form for p in results] == [
        "string",
        "list",
        "absent",
        "string",
        "string",
        "string",
    ]
    assert roundtrip(native) == native


def _pdf(pages: int) -> str:
    body = b"%%PDF-1.7\n1 0 obj << /Type /Pages /Count %d >> endobj\n" % pages
    body += b"".join(b"%d 0 obj << /Type /Page >> endobj\n" % (i + 2) for i in range(pages))
    return base64.b64encode(body).decode()


PDF = {
    "type": "document",
    "source": {"type": "base64", "media_type": "application/pdf", "data": _pdf(3)},
}


@pytest.mark.parametrize(
    ("block", "expected"),
    [
        (
            {"type": "thinking", "thinking": "hmm", "signature": "sig=="},
            {"kind": "thinking", "text": "hmm"},
        ),
        ({"type": "redacted_thinking", "data": "ZW5jcnlwdGVk"}, {"kind": "thinking"}),
        ({"type": "image", "source": {"type": "url", "url": "https://x/y.png"}}, {"kind": "image"}),
        (
            {
                "type": "document",
                "source": {"type": "text", "media_type": "text/plain", "data": "t"},
            },
            {"kind": "document", "text": "t"},
        ),
        (PDF, {"kind": "document", "pages": 3}),
        (
            {
                "type": "document",
                "source": {"type": "base64", "media_type": "application/pdf", "data": "!!"},
            },
            {"kind": "document"},
        ),
        (
            {"type": "server_tool_use", "id": "s1", "name": "web_search", "input": {}},
            {"kind": "unknown"},
        ),
    ],
)
def test_blocks_the_core_does_not_handle_become_opaque_with_measures_and_come_back_unchanged(
    block: dict[str, Any], expected: dict[str, Any]
) -> None:
    native = [{"role": "assistant", "content": [block]}]
    part = anthropic.to_messages(native)[0].parts[0]
    assert part == Opaque(raw=block, **expected)
    content = anthropic.from_messages(anthropic.to_messages(native))[0]["content"]
    assert isinstance(content, list)
    assert content[0] is block
    assert roundtrip(native) == native


@pytest.mark.parametrize(
    "native",
    [
        [{"role": "system", "content": "x"}],
        [{"role": "user"}],
        [{"role": "user", "content": 3}],
        [{"role": "user", "content": [{"text": "no type"}]}],
        [
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "t", "name": "n", "input": [1]}],
            }
        ],
        ["not a dict"],
    ],
)
def test_input_outside_the_supported_range_is_refused(native: list[Any]) -> None:
    with pytest.raises(UnsupportedInput):
        anthropic.to_messages(native)


def test_a_neutral_message_built_by_hand_converts_to_blocks() -> None:
    msg = Message(
        "assistant",
        (Text("a"), ToolCall(id="t1", name="n", raw_args='{"k": 1}')),
    )
    assert anthropic.from_messages([msg]) == [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "a"},
                {"type": "tool_use", "id": "t1", "name": "n", "input": {"k": 1}},
            ],
        }
    ]


# --- property: any conversation in the supported range round-trips ---

json_scalars = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(),
    st.floats(allow_nan=False, allow_infinity=False),
    st.text(max_size=8),
)
json_values = st.recursive(
    json_scalars,
    lambda inner: st.one_of(
        st.lists(inner, max_size=3), st.dictionaries(st.text(max_size=5), inner, max_size=3)
    ),
    max_leaves=6,
)
ids = st.text(alphabet="abcdef0123456789", min_size=1, max_size=6)
maybe_cache: st.SearchStrategy[dict[str, Any]] = st.one_of(
    st.just({}), st.just({"cache_control": CACHE})
)

text_block = st.builds(
    lambda t, c: {"type": "text", "text": t, **c}, st.text(max_size=20), maybe_cache
)
image_block = st.just({"type": "image", "source": {"type": "url", "url": "https://x/i.png"}})
thinking_block = st.builds(
    lambda t: {"type": "thinking", "thinking": t, "signature": "sig"}, st.text(max_size=10)
)
other_block = st.sampled_from(
    [
        {"type": "redacted_thinking", "data": "ZW5j"},
        {"type": "document", "source": {"type": "text", "media_type": "text/plain", "data": "d"}},
        PDF,
        {"type": "server_tool_use", "id": "s1", "name": "web_search", "input": {}},
    ]
)
message_extra: st.SearchStrategy[dict[str, Any]] = st.one_of(
    st.just({}), st.just({"cache_control": CACHE}), st.just({"x-trace": "t1"})
)
tool_use_block = st.builds(
    lambda i, n, inp, c: {"type": "tool_use", "id": i, "name": n, "input": inp, **c},
    ids,
    st.text(min_size=1, max_size=6),
    st.dictionaries(st.text(max_size=5), json_values, max_size=3),
    st.one_of(maybe_cache, st.just({"caller": {"type": "direct"}})),
)


@st.composite
def tool_result_block(draw: st.DrawFn) -> dict[str, Any]:
    block: dict[str, Any] = {"type": "tool_result", "tool_use_id": draw(ids)}
    form = draw(st.sampled_from(["str", "list", "absent"]))
    if form == "str":
        block["content"] = draw(st.text(max_size=20))
    elif form == "list":
        block["content"] = draw(
            st.lists(st.one_of(text_block, image_block, other_block), max_size=3)
        )
    if draw(st.booleans()):
        block["is_error"] = draw(st.booleans())
    block.update(draw(maybe_cache))
    return block


user_message = st.builds(
    lambda c, x: {"role": "user", "content": c, **x},
    st.one_of(
        st.text(max_size=20),
        st.lists(st.one_of(text_block, image_block, other_block, tool_result_block()), max_size=4),
    ),
    message_extra,
)
assistant_message = st.builds(
    lambda c, x: {"role": "assistant", "content": c, **x},
    st.one_of(
        st.text(max_size=20),
        st.lists(st.one_of(text_block, thinking_block, other_block, tool_use_block), max_size=4),
    ),
    message_extra,
)


@given(st.lists(st.one_of(user_message, assistant_message), max_size=6))
def test_every_supported_conversation_round_trips(native: list[dict[str, Any]]) -> None:
    assert roundtrip(native) == native


@pytest.mark.parametrize(
    "args",
    [{1: "a"}, {"a": (1, 2)}, {"a": float("nan")}, {"a": float("inf")}],
    ids=["int-key", "tuple", "nan", "inf"],
)
def test_tool_use_input_that_would_not_come_back_equal_is_refused(args: dict[Any, Any]) -> None:
    native = [
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "t", "name": "n", "input": args}],
        }
    ]
    with pytest.raises(UnsupportedInput, match="input"):
        anthropic.to_messages(native)
