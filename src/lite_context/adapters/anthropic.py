"""Anthropic Messages API `messages` arrays as plain dicts (design §3.2). No SDK import.

Supported: roles user / assistant; string content or a block list; `text`, `tool_use` and
`tool_result` blocks (string, block-list or absent content; `is_error`); any other block becomes
`Opaque` and comes back as the same object. The system prompt is a separate parameter in this
API, so it never appears here.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

from ..errors import UnsupportedInput
from ..types import ContentForm, Message, Opaque, Part, Text, ToolCall, ToolResult
from ._common import (
    content,
    mapping,
    native_content,
    native_text,
    opaque,
    rest,
    restore,
    string,
    text_part,
)


def to_messages(native: Sequence[object]) -> list[Message]:
    return [_message(m, f"message {i}") for i, m in enumerate(native)]


def from_messages(messages: Sequence[Message]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for m in messages:
        core: dict[str, object] = {
            "role": m.role,
            "content": native_content(m.parts, m.content_form, _block),
        }
        out.append(restore(core, m.extra))
    return out


def _message(value: object, where: str) -> Message:
    msg = mapping(value, where)
    role = msg.get("role")
    if role not in ("user", "assistant"):
        raise UnsupportedInput(f"{where}: role {role!r} is not user or assistant")
    if "content" not in msg:
        raise UnsupportedInput(f"{where}: no content")
    parts, form = content(msg["content"], where, _part)
    return Message(role, parts, content_form=form, extra=rest(msg, "role", "content"))


def _part(value: object, where: str) -> Part:
    block = mapping(value, where)
    kind = string(block.get("type"), f"{where} type")
    if kind == "text":
        return text_part(block, where)
    if kind == "tool_use":
        args = mapping(block.get("input"), f"{where} input")
        try:
            raw_args = json.dumps(args, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise UnsupportedInput(f"{where}: tool_use input is not JSON: {exc}") from exc
        if json.loads(raw_args) != args:  # int keys, tuples ...: would not come back equal
            raise UnsupportedInput(f"{where}: tool_use input is not plain JSON")
        return ToolCall(
            id=string(block.get("id"), f"{where} id"),
            name=string(block.get("name"), f"{where} name"),
            raw_args=raw_args,
            extra=rest(block, "type", "id", "name", "input"),
        )
    if kind == "tool_result":
        return _tool_result(block, where)
    return _opaque(block)


def _opaque(block: Mapping[str, object]) -> Opaque:
    kind = block.get("type")
    if kind == "image":
        return opaque(block, "image")
    if kind == "document":
        source = block.get("source")
        source = source if isinstance(source, Mapping) else {}
        data = source.get("data")
        if source.get("type") == "text":
            return opaque(block, "document", text=data)
        if source.get("media_type") == "application/pdf":
            return opaque(block, "document", pdf=data)
        return opaque(block, "document")
    if kind == "thinking":
        return opaque(block, "thinking", text=block.get("thinking"))
    if kind == "redacted_thinking":
        return opaque(block, "thinking")
    return opaque(block, "unknown")


def _result_part(value: object, where: str) -> Text | Opaque:
    block = mapping(value, where)
    return text_part(block, where) if block.get("type") == "text" else _opaque(block)


def _tool_result(block: Mapping[str, object], where: str) -> ToolResult:
    parts: tuple[Text | Opaque, ...] = ()
    form: ContentForm = "absent"
    if "content" in block:
        parts, form = content(block["content"], where, _result_part)
    is_error = block.get("is_error")
    if is_error is not None and not isinstance(is_error, bool):
        raise UnsupportedInput(f"{where}: is_error must be a boolean")
    return ToolResult(
        call_id=string(block.get("tool_use_id"), f"{where} tool_use_id"),
        content=parts,
        is_error=is_error,
        content_form=form,
        extra=rest(block, "type", "tool_use_id", "content", "is_error"),
    )


def _block(part: Part) -> object:
    if isinstance(part, Text):
        return native_text(part)
    if isinstance(part, ToolCall):
        try:
            args = part.args()
        except ValueError as exc:  # json.JSONDecodeError is a ValueError
            raise UnsupportedInput(
                f"tool call {part.id}: arguments are not a JSON object; Anthropic needs one"
            ) from exc
        core: dict[str, object] = {"type": "tool_use", "id": part.id, "name": part.name}
        core["input"] = args
        return restore(core, part.extra)
    if isinstance(part, ToolResult):
        return _native_result(part)
    return part.raw


def _native_result(part: ToolResult) -> dict[str, object]:
    core: dict[str, object] = {"type": "tool_result", "tool_use_id": part.call_id}
    if part.content or part.content_form == "list":
        core["content"] = native_content(part.content, part.content_form, _block)
    if part.is_error is not None:
        core["is_error"] = part.is_error
    return restore(core, part.extra)
