"""OpenAI Chat Completions `messages` lists as plain dicts (design §3.2). No SDK import.

Supported: leading `system` / `developer` messages (returned as `fixed`, never cut, put back in
front); `user` with string or part-list content; `assistant` with string, part-list, null or
absent content and `tool_calls` (`arguments` kept verbatim, malformed or not); `tool` messages,
each its own neutral user message holding one `ToolResult`. A system message after the
conversation has started, and the legacy `function` role, are refused. OpenAI has no error flag on
a tool result, so `ToolResult.is_error` does not reach the output.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

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

_SYSTEM_ROLES = ("system", "developer")
_PDF_DATA_URL = "data:application/pdf;base64,"


@dataclass(frozen=True)
class Converted:
    fixed: tuple[Mapping[str, object], ...]  # the leading system / developer messages, verbatim
    messages: list[Message]

    def fixed_texts(self) -> tuple[str, ...]:
        """The fixed messages' text, for `Budget.available` / `fit(fixed=...)`."""
        texts = []
        for msg in self.fixed:
            value = msg.get("content")
            if isinstance(value, list):
                value = "\n".join(
                    p["text"]
                    for p in value
                    if isinstance(p, Mapping) and isinstance(p.get("text"), str)
                )
            texts.append(value if isinstance(value, str) else "")
        return tuple(texts)


def to_messages(native: Sequence[object]) -> Converted:
    fixed: list[Mapping[str, object]] = []
    messages: list[Message] = []
    for i, value in enumerate(native):
        where = f"message {i}"
        msg = mapping(value, where)
        role = msg.get("role")
        if role in _SYSTEM_ROLES:
            if messages:
                raise UnsupportedInput(f"{where}: a {role} message after the conversation started")
            fixed.append(msg)
        elif role == "user":
            parts, form = _required_content(msg, where)
            messages.append(
                Message("user", parts, content_form=form, extra=rest(msg, "role", "content"))
            )
        elif role == "assistant":
            messages.append(_assistant(msg, where))
        elif role == "tool":
            messages.append(_tool(msg, where))
        else:
            raise UnsupportedInput(f"{where}: role {role!r} is not supported")
    return Converted(tuple(fixed), messages)


def from_messages(
    messages: Sequence[Message], fixed: Sequence[Mapping[str, object]] = ()
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = [dict(f) for f in fixed]
    for message in messages:
        if message.role == "assistant":
            out.append(_native_assistant(message))
        else:
            out.extend(_native_user(message))
    return out


# --- to neutral ---


def _content_part(value: object, where: str) -> Text | Opaque:
    part = mapping(value, where)
    kind = string(part.get("type"), f"{where} type")
    if kind == "text":
        return text_part(part, where)
    if kind == "image_url":
        return opaque(part, "image")
    if kind == "file":
        file = part.get("file")
        data = file.get("file_data") if isinstance(file, Mapping) else None
        if isinstance(data, str) and data.startswith(_PDF_DATA_URL):
            return opaque(part, "document", pdf=data[len(_PDF_DATA_URL) :])
        return opaque(part, "document")
    return opaque(part, "unknown")


def _required_content(
    msg: Mapping[str, object], where: str
) -> tuple[tuple[Text | Opaque, ...], ContentForm]:
    if msg.get("content") is None:
        raise UnsupportedInput(f"{where}: needs content")
    return content(msg["content"], where, _content_part)


def _assistant(msg: Mapping[str, object], where: str) -> Message:
    parts: tuple[Part, ...] = ()
    form: ContentForm = "absent"
    if "content" in msg:
        if msg["content"] is None:
            form = "null"
        else:
            parts, form = content(msg["content"], where, _content_part)
    raw_calls = msg.get("tool_calls") or []  # absent, None (the SDK's model_dump) or []
    if not isinstance(raw_calls, list):
        raise UnsupportedInput(f"{where}: tool_calls must be a list")
    calls = tuple(_call(c, f"{where} tool call {j}") for j, c in enumerate(raw_calls))
    # `tool_calls: []` / `None` stay in extra, verbatim; real calls are parts.
    taken = ("role", "content", "tool_calls") if calls else ("role", "content")
    return Message("assistant", (*parts, *calls), content_form=form, extra=rest(msg, *taken))


def _call(value: object, where: str) -> ToolCall:
    call = mapping(value, where)
    if call.get("type") != "function":
        raise UnsupportedInput(f"{where}: type {call.get('type')!r} is not a function call")
    function = mapping(call.get("function"), f"{where} function")
    extra = rest(call, "id", "function")
    function_extra = rest(function, "name", "arguments")
    if function_extra:
        extra["function"] = function_extra
    return ToolCall(
        id=string(call.get("id"), f"{where} id"),
        name=string(function.get("name"), f"{where} name"),
        raw_args=string(function.get("arguments"), f"{where} arguments"),
        extra=extra,
    )


def _tool(msg: Mapping[str, object], where: str) -> Message:
    parts, form = _required_content(msg, where)
    result = ToolResult(
        call_id=string(msg.get("tool_call_id"), f"{where} tool_call_id"),
        content=parts,
        content_form=form,
        extra=rest(msg, "role", "tool_call_id", "content"),
    )
    return Message("user", (result,))


# --- to native ---


def _render(part: Part) -> object:
    if isinstance(part, Text):
        return native_text(part)
    if isinstance(part, Opaque):
        return part.raw
    raise UnsupportedInput(f"a {type(part).__name__} cannot be message content in this format")


def _native_user(message: Message) -> list[dict[str, object]]:
    """Each tool result becomes a tool message and each run of other parts a user message, in
    the order the parts come."""
    out: list[dict[str, object]] = []
    run: list[Part] = []

    def flush() -> None:
        content = native_content(run, message.content_form, _render)
        out.append(restore({"role": "user", "content": content}, message.extra))
        run.clear()

    for part in message.parts:
        if isinstance(part, ToolResult):
            if run:
                flush()
            core: dict[str, object] = {
                "role": "tool",
                "tool_call_id": part.call_id,
                "content": native_content(part.content, part.content_form, _render),
            }
            out.append(restore(core, part.extra))
        else:
            run.append(part)
    if run or not out:
        flush()
    return out


def _native_assistant(message: Message) -> dict[str, object]:
    calls = [p for p in message.parts if isinstance(p, ToolCall)]
    others = [p for p in message.parts if not isinstance(p, ToolCall)]
    form = message.content_form
    core: dict[str, object] = {"role": "assistant"}
    if others or form in ("string", "list"):
        core["content"] = native_content(others, form, _render)
    elif form != "absent":
        core["content"] = None  # "null", or built by hand with only tool calls
    if calls:
        core["tool_calls"] = [_native_call(c) for c in calls]
    return restore(core, message.extra)


def _native_call(call: ToolCall) -> dict[str, object]:
    function = call.extra.get("function")
    core: dict[str, object] = {"id": call.id, "type": "function"}
    core["function"] = restore(
        {"name": call.name, "arguments": call.raw_args},
        function if isinstance(function, Mapping) else {},
    )
    return restore(core, call.extra)
