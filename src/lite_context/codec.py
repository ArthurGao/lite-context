"""Neutral messages and states as JSON-ready records, and back (design §8): what a `Transcript`
store keeps.

Every field of every part is kept, so `message_from_dict(message_to_dict(m)) == m`. A record is
checked when it is written exactly as it will be when read: a transcript is append-only, so a
record that could not be read back would make it unreadable for good. Refused when written:
- values JSON would not give back equal (`Opaque.raw`, the `extra` mappings): a tuple, a
  non-string key, NaN, an object;
- fields of the wrong type (`pinned=1`, `tokens=True`, a tool result inside a tool result);
- a NUL character or a lone surrogate anywhere, which PostgreSQL cannot store: refused by every
  store, so they all behave alike.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any, Literal, cast, get_args

from .state import ContextState, _str
from .types import ContentForm, Message, Opaque, OpaqueKind, Part, Role, Text, ToolCall, ToolResult

VERSION = 1

Kind = Literal["message", "state"]

_ROLES = get_args(Role)
_KINDS = get_args(OpaqueKind)
_FORMS = (None, *get_args(ContentForm))
_UNSTORABLE = re.compile("[\x00\ud800-\udfff]")


def message_to_dict(m: Message) -> dict[str, object]:
    record: dict[str, object] = {
        "v": VERSION,
        "role": m.role,
        "parts": [_part(p) for p in m.parts],
        "key": m.key,
        "pinned": m.pinned,
        "content_form": m.content_form,
        "extra": _json(m.extra, "message extra"),
    }
    _storable(record)
    message_from_dict(record)  # what is written can be read
    return record


def message_from_dict(data: Mapping[str, Any]) -> Message:
    if data.get("v") != VERSION:
        raise ValueError(f"unknown message record version {data.get('v')!r}")
    try:
        return Message(
            role=cast(Role, _one_of(data["role"], _ROLES)),
            parts=tuple(_back(p) for p in data["parts"]),
            key=_typed(data["key"], str, optional=True),
            pinned=_typed(data["pinned"], bool),
            content_form=cast("ContentForm | None", _one_of(data["content_form"], _FORMS)),
            extra=_typed(data["extra"], dict),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"not a message record: {exc}") from exc


def item_to_record(item: Message | ContextState) -> tuple[Kind, dict[str, Any]]:
    """The kind and the JSON-ready data a store keeps for a message or a state."""
    if isinstance(item, ContextState):
        return "state", json.loads(item.to_json())
    return "message", message_to_dict(item)


def record_to_item(kind: str, data: Mapping[str, Any]) -> Message | ContextState:
    if kind == "state":
        return ContextState.from_json(json.dumps(data))
    if kind == "message":
        return message_from_dict(data)
    raise ValueError(f"unknown record kind {kind!r}")


def _part(p: Part) -> dict[str, object]:
    if isinstance(p, Text):
        return {"type": "text", "text": p.text, "extra": _json(p.extra, "text extra")}
    if isinstance(p, ToolCall):
        return {
            "type": "tool_call",
            "id": p.id,
            "name": p.name,
            "raw_args": p.raw_args,
            "extra": _json(p.extra, "tool call extra"),
        }
    if isinstance(p, ToolResult):
        return {
            "type": "tool_result",
            "call_id": p.call_id,
            "content": [_part(c) for c in p.content],
            "is_error": p.is_error,
            "content_form": p.content_form,
            "extra": _json(p.extra, "tool result extra"),
        }
    return {
        "type": "opaque",
        "kind": p.kind,
        "raw": _json(p.raw, f"a {p.kind} part's raw value"),
        "tokens": p.tokens,
        "text": p.text,
        "pages": p.pages,
    }


def _back(d: Mapping[str, Any]) -> Part:
    kind = d["type"]
    if kind == "text":
        return Text(_str(d["text"]), extra=_typed(d["extra"], dict))
    if kind == "tool_call":
        return ToolCall(
            _str(d["id"]), _str(d["name"]), _str(d["raw_args"]), extra=_typed(d["extra"], dict)
        )
    if kind == "tool_result":
        content = tuple(_back(c) for c in d["content"])
        if not all(isinstance(c, Text | Opaque) for c in content):
            raise ValueError("a tool result holds only text and opaque parts")
        return ToolResult(
            _str(d["call_id"]),
            cast("tuple[Text | Opaque, ...]", content),
            is_error=_typed(d["is_error"], bool, optional=True),
            content_form=cast("ContentForm | None", _one_of(d["content_form"], _FORMS)),
            extra=_typed(d["extra"], dict),
        )
    if kind == "opaque":
        return Opaque(
            cast(OpaqueKind, _one_of(d["kind"], _KINDS)),
            d["raw"],
            tokens=_typed(d["tokens"], int, optional=True),
            text=_typed(d["text"], str, optional=True),
            pages=_typed(d["pages"], int, optional=True),
        )
    raise ValueError(f"unknown part type {kind!r}")


def _json(value: Any, what: str) -> Any:
    """`value` itself when JSON gives it back equal; otherwise `ValueError`."""
    try:
        text = json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{what} is not JSON: {exc}") from exc
    if json.loads(text) != value:
        raise ValueError(f"{what} is not JSON that comes back equal (a tuple or a non-string key?)")
    return value


def _storable(value: Any) -> None:
    """Every string in `value`, keys included, free of NUL and lone surrogates."""
    if isinstance(value, str):
        if _UNSTORABLE.search(value):
            raise ValueError("a NUL character or a lone surrogate cannot be stored")
    elif isinstance(value, dict):
        for k, v in value.items():
            _storable(k)
            _storable(v)
    elif isinstance(value, list):
        for v in value:
            _storable(v)


def _one_of(value: Any, allowed: tuple[object, ...]) -> Any:
    if value not in allowed:
        raise ValueError(f"{value!r} is not one of {allowed}")
    return value


def _typed(value: Any, kind: type, *, optional: bool = False) -> Any:
    """`value` when it is a `kind` (a bool is not an int here), or None when `optional`."""
    if value is None and optional:
        return None
    if not isinstance(value, kind) or (kind is int and isinstance(value, bool)):
        raise TypeError(f"expected {kind.__name__}, got {type(value).__name__}")
    return value


__all__ = [
    "Kind",
    "item_to_record",
    "message_from_dict",
    "message_to_dict",
    "record_to_item",
]
