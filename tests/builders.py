"""Tiny conversation builders for tests. With `COUNTER` one character is one token and a
message costs nothing extra, so expected numbers can be read off the text."""

from lite_context import Message, Opaque, Text, ToolCall, ToolResult
from lite_context.testing import COUNTER

__all__ = ["COUNTER", "calls", "human", "image", "results", "say", "thinking_call"]


def human(key: str, text: str, *, pinned: bool = False) -> Message:
    return Message("user", (Text(text),), key=key, pinned=pinned)


def say(key: str, text: str) -> Message:
    return Message("assistant", (Text(text),), key=key)


def calls(key: str, *ids: str, name: str = "search") -> Message:
    return Message(
        "assistant", tuple(ToolCall(id=i, name=name, raw_args="{}") for i in ids), key=key
    )


def results(key: str, *pairs: tuple[str, str], pinned: bool = False) -> Message:
    return Message(
        "user",
        tuple(ToolResult(call_id=i, content=(Text(t),)) for i, t in pairs),
        key=key,
        pinned=pinned,
    )


def image(key: str, text: str = "look") -> Message:
    return Message(
        "user", (Text(text), Opaque(kind="image", raw={"type": "image", "k": key})), key=key
    )


def thinking_call(key: str, call_id: str, thought: str) -> Message:
    return Message(
        "assistant",
        (
            Opaque(kind="thinking", raw={"type": "thinking", "thinking": thought}, text=thought),
            ToolCall(id=call_id, name="search", raw_args="{}"),
        ),
        key=key,
    )
