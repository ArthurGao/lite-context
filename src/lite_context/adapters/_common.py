"""Helpers shared by the adapters: input checks, the text / opaque part shapes both vendors
share, content forms, and the neutral measures of opaque content (design §3.1)."""

from __future__ import annotations

import base64
import binascii
import re
from collections.abc import Callable, Mapping, Sequence
from typing import TypeVar

from ..errors import UnsupportedInput
from ..types import ContentForm, Opaque, OpaqueKind, Part, Text

_PDF_PAGE = re.compile(rb"/Type\s*/Page(?![a-zA-Z])")
P = TypeVar("P", bound=Part)


def mapping(value: object, what: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise UnsupportedInput(f"{what} must be an object, got {type(value).__name__}")
    return value


def string(value: object, what: str) -> str:
    if not isinstance(value, str):
        raise UnsupportedInput(f"{what} must be a string, got {type(value).__name__}")
    return value


def rest(source: Mapping[str, object], *taken: str) -> dict[str, object]:
    """The fields of `source` the core does not model, kept to be restored verbatim."""
    return {k: v for k, v in source.items() if k not in taken}


def restore(core: dict[str, object], extra: Mapping[str, object]) -> dict[str, object]:
    """`core` fields first, then the kept fields that do not collide with them."""
    for k, v in extra.items():
        core.setdefault(k, v)
    return core


def text_part(block: Mapping[str, object], where: str) -> Text:
    return Text(string(block.get("text"), f"{where} text"), extra=rest(block, "type", "text"))


def native_text(part: Text) -> dict[str, object]:
    return restore({"type": "text", "text": part.text}, part.extra)


def content(
    value: object, where: str, part: Callable[[object, str], P]
) -> tuple[tuple[Text | P, ...], ContentForm]:
    """Parse a content value that is a string or a list of parts."""
    if isinstance(value, str):
        return (Text(value),), "string"
    if isinstance(value, list):
        return tuple(part(p, f"{where} part {j}") for j, p in enumerate(value)), "list"
    raise UnsupportedInput(f"{where}: content must be a string or a list")


def native_content(
    parts: Sequence[Part], form: ContentForm | None, render: Callable[[Part], object]
) -> object:
    """A string when the content was one (or, built by hand with no recorded form, is) a single
    plain text; otherwise a list."""
    if form != "list" and len(parts) == 1 and isinstance(parts[0], Text) and not parts[0].extra:
        return parts[0].text
    return [render(p) for p in parts]


def pdf_pages(b64: str) -> int | None:
    """Pages of a base64 PDF (counted once, at conversion), or None when it cannot be read."""
    try:
        body = base64.b64decode(b64, validate=False)
    except (binascii.Error, ValueError):
        return None
    return len(_PDF_PAGE.findall(body)) or None


def opaque(raw: object, kind: OpaqueKind, *, text: object = None, pdf: object = None) -> Opaque:
    """An opaque part with the neutral measures the counter needs: its own text (thinking, a
    text document) or a PDF's page count."""
    return Opaque(
        kind=kind,
        raw=raw,
        text=text if isinstance(text, str) else None,
        pages=pdf_pages(pdf) if isinstance(pdf, str) else None,
    )
