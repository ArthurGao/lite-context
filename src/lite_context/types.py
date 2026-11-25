"""Neutral message types (design §3.1). Adapters convert vendor formats to and from these."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Literal, TypeAlias

Role: TypeAlias = Literal["user", "assistant"]
OpaqueKind: TypeAlias = Literal["image", "document", "thinking", "unknown"]
# The form a vendor's content had, so an adapter can give it back (design §3.2). None: built by
# hand; a single plain text then goes out as a string, anything else as a list.
ContentForm: TypeAlias = Literal["string", "list", "null", "absent"]

_ROLES = ("user", "assistant")


@dataclass(frozen=True)
class Text:
    text: str
    extra: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolCall:
    """A tool call. `raw_args` is kept verbatim: a model may emit malformed JSON."""

    id: str
    name: str
    raw_args: str
    extra: Mapping[str, object] = field(default_factory=dict)

    def args(self) -> dict[str, object]:
        """Parse `raw_args`; raises `json.JSONDecodeError` or `ValueError`."""
        value = json.loads(self.raw_args)
        if not isinstance(value, dict):
            raise ValueError(f"tool call {self.id}: arguments are not a JSON object")
        return value


@dataclass(frozen=True)
class Opaque:
    """Content the core carries but never reads: images, documents, thinking, unknown blocks.

    `raw` is the original object and goes back out unchanged. `tokens` is its cost when the caller
    knows it; otherwise the counter estimates it (design §4) from `kind` and the neutral measures
    the adapter took at conversion: `text` (a thinking block's or a text document's own text) and
    `pages` (a PDF's page count). The counter never reads `raw`.
    """

    kind: OpaqueKind
    raw: object
    tokens: int | None = None
    text: str | None = None
    pages: int | None = None


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    content: tuple[Text | Opaque, ...]
    is_error: bool | None = None  # None: the vendor format had no error flag on it
    content_form: ContentForm | None = None
    extra: Mapping[str, object] = field(default_factory=dict)


Part: TypeAlias = Text | ToolCall | ToolResult | Opaque


@dataclass(frozen=True)
class Message:
    """One message. `key` is the caller's stable id; `pinned` messages are never cut (design §5.2).

    `extra` carries fields the core ignores so an adapter can restore them verbatim.
    """

    role: Role
    parts: tuple[Part, ...]
    key: str | None = None
    pinned: bool = False
    content_form: ContentForm | None = None
    extra: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.role not in _ROLES:
            raise ValueError(f"role must be one of {_ROLES}, got {self.role!r}")


def with_keys(messages: Sequence[Message], keys: Iterable[str]) -> list[Message]:
    """Return copies of `messages` carrying `keys`, one per message; the input is untouched."""
    keys = list(keys)
    if len(keys) != len(messages):
        raise ValueError(f"{len(messages)} messages but {len(keys)} keys")
    check_unique(keys)
    return [replace(m, key=k) for m, k in zip(messages, keys, strict=True)]


def check_unique(keys: Iterable[str]) -> None:
    seen: set[str] = set()
    for key in keys:
        if key in seen:
            raise ValueError(f"duplicate key {key!r}")
        seen.add(key)
