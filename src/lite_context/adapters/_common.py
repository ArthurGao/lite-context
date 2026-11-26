"""Helpers shared by the adapters: input checks, the text / opaque part shapes both vendors
share, content forms, and the neutral measures of opaque content (design §3.1)."""

from __future__ import annotations

import base64
import binascii
import re
import zlib
from collections.abc import Callable, Mapping, Sequence
from typing import TypeVar

from ..errors import UnsupportedInput
from ..types import ContentForm, Opaque, OpaqueKind, Part, Text

_PDF_PAGE = re.compile(rb"/Type\s*/Page(?![a-zA-Z])")
_PDF_PAGES = re.compile(rb"/Type\s*/Pages(?![a-zA-Z])")
_PDF_COUNT = re.compile(rb"/Count\s+(\d+)")
# PDF 1.5+ often keeps the page objects in compressed object streams; reading them is bounded so
# a small file that inflates to gigabytes (a decompression bomb) costs at most this much.
MAX_INFLATED = 64 * 1024 * 1024
# A page-tree node is a small dictionary: it is looked for this far around its /Type, and only
# for this many nodes, so no input makes the search quadratic.
_NODE_REACH = 4096
_MAX_NODES = 1000
_DELIMITERS = {b" ", b"\t", b"\r", b"\n", b"\x0c", b"\x00", b">"}
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
    """Pages of a base64 PDF (counted once, at conversion), or None when it cannot be read: the
    page objects in the file, else in its compressed streams, else the page tree's /Count.
    Every step is linear in the file (and bounded in the inflated data)."""
    try:
        body = base64.b64decode(b64, validate=False)
    except (binascii.Error, ValueError):
        return None
    inflated = bytearray()
    pages = _visible_pages(body) or _visible_pages(inflated := _inflated(body))
    if pages:
        return pages
    counts = [c for blob in (body, inflated) for c in _tree_counts(blob)]
    return max(counts) if counts else None


def _visible_pages(blob: bytes | bytearray) -> int:
    return len(_PDF_PAGE.findall(blob))


def _inflated(body: bytes) -> bytearray:
    """The FlateDecode streams of a PDF, decompressed, at most `MAX_INFLATED` bytes in all; a
    stream that is not zlib data is skipped. Each `stream` keyword (after a delimiter, before an
    end of line) is followed to the next `endstream`, whether or not a line break comes first."""
    out = bytearray()
    at = 0
    while len(out) < MAX_INFLATED:
        start = body.find(b"stream", at)
        if start == -1:
            break
        at = start + 6
        if start and body[start - 1 : start] not in _DELIMITERS:
            continue  # inside a word: `endstream`, `upstream`
        if body.startswith(b"\r\n", at):
            at += 2
        elif body.startswith(b"\n", at):
            at += 1
        else:
            continue  # the word inside text, not the keyword
        end = body.find(b"endstream", at)
        if end == -1:
            break  # nothing after this one ends either
        try:
            out += zlib.decompressobj().decompress(body[at:end], MAX_INFLATED - len(out))
        except zlib.error:
            pass
        at = end + 9
    return out


def _tree_counts(blob: bytes | bytearray) -> list[int]:
    """The /Count of each page-tree node (a dictionary holding /Type /Pages), read from that
    dictionary's own entries: a /Count inside a nested dictionary (/Resources) is not it."""
    counts: list[int] = []
    for n, match in enumerate(_PDF_PAGES.finditer(blob)):
        if n == _MAX_NODES:
            break
        counts += _node_counts(blob, match.start())
    return counts


def _node_counts(blob: bytes | bytearray, pos: int) -> list[int]:
    """The /Count entries of the innermost `<< ... >>` holding `pos`, found within
    `_NODE_REACH` bytes. Each search resumes where the last one stopped, so a node costs one
    pass over its window however many marks it holds."""
    low, high = max(pos - _NODE_REACH, 0), min(pos + _NODE_REACH, len(blob))
    # Backward to the `<<` that opens it, skipping dictionaries closed before `pos`.
    opened, closed, depth = blob.rfind(b"<<", low, pos), blob.rfind(b">>", low, pos), 0
    while True:
        if opened == -1:
            return []
        if closed > opened:
            depth += 1
            closed = blob.rfind(b">>", low, closed)
        elif depth:
            depth -= 1  # this `<<` opened a dictionary that closed before `pos`
            opened = blob.rfind(b"<<", low, opened)
        else:
            break
    # Forward to its `>>`, reading /Count only where the depth is the node's own.
    counts: list[int] = []
    segment, depth, cursor = opened + 2, 0, opened + 2
    nxt_open, nxt_close = blob.find(b"<<", cursor, high), blob.find(b">>", cursor, high)
    while True:
        if nxt_close == -1:
            return []
        if nxt_open != -1 and nxt_open < nxt_close:
            if depth == 0:
                counts += [int(c) for c in _PDF_COUNT.findall(blob, segment, nxt_open)]
            depth += 1
            nxt_open = blob.find(b"<<", nxt_open + 2, high)
        elif depth:
            depth -= 1
            if depth == 0:
                segment = nxt_close + 2
            nxt_close = blob.find(b">>", nxt_close + 2, high)
        else:
            counts += [int(c) for c in _PDF_COUNT.findall(blob, segment, nxt_close)]
            return counts


def opaque(raw: object, kind: OpaqueKind, *, text: object = None, pdf: object = None) -> Opaque:
    """An opaque part with the neutral measures the counter needs: its own text (thinking, a
    text document) or a PDF's page count."""
    return Opaque(
        kind=kind,
        raw=raw,
        text=text if isinstance(text, str) else None,
        pages=pdf_pages(pdf) if isinstance(pdf, str) else None,
    )
