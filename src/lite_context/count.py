"""Token counting and budgets (design §4). Counts are estimates used to decide; the vendor's
reported usage is what a host bills by."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

from .errors import DoesNotFit
from .report import Report
from .types import Message, Opaque, Part, Text, ToolCall, ToolResult


class Counter(Protocol):
    @property
    def per_message(self) -> int:
        """Framing overhead per message."""
        ...

    def text(self, s: str) -> int: ...

    def opaque(self, part: Opaque) -> int:
        """The cost of an opaque part whose `tokens` the caller did not give."""
        ...


@dataclass(frozen=True)
class CharEstimate:
    """Characters / `chars_per_token`, rounded up. Opaque parts are estimated from their kind and
    the measures the adapter took (design §3.1): never from `raw`."""

    chars_per_token: float = 3.5
    per_message: int = 4
    image_tokens: int = 1600
    pdf_page_tokens: int = 2000
    opaque_default: int = 1500

    def __post_init__(self) -> None:
        if self.chars_per_token <= 0:
            raise ValueError(f"chars_per_token must be positive, got {self.chars_per_token}")
        for name in ("per_message", "image_tokens", "pdf_page_tokens", "opaque_default"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must not be negative")

    def text(self, s: str) -> int:
        return math.ceil(len(s) / self.chars_per_token)

    def opaque(self, part: Opaque) -> int:
        if part.kind == "image":
            return self.image_tokens
        if part.pages is not None:
            return part.pages * self.pdf_page_tokens
        if part.text is not None:
            return self.text(part.text)
        return self.opaque_default


def _count_part(part: Part, counter: Counter) -> int:
    if isinstance(part, Text):
        return counter.text(part.text)
    if isinstance(part, ToolCall):
        return counter.text(part.name) + counter.text(part.raw_args)
    if isinstance(part, ToolResult):
        return sum(_count_part(c, counter) for c in part.content)
    return part.tokens if part.tokens is not None else counter.opaque(part)


def longest_prefix(text: str, tokens: int, counter: Counter, *, from_end: bool = False) -> str:
    """The longest prefix (or suffix) of `text` that costs at most `tokens`, by binary search over
    a window sized from the text's own characters per token, widened while it all fits."""
    total = counter.text(text)
    if total <= tokens:
        return text
    window = max(1, min(len(text), 2 * tokens * len(text) // max(total, 1) + 1))
    while window < len(text) and counter.text(_piece(text, window, from_end)) <= tokens:
        window = min(len(text), window * 2)
    lo, hi = 0, window
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if counter.text(_piece(text, mid, from_end)) <= tokens:
            lo = mid
        else:
            hi = mid - 1
    return _piece(text, lo, from_end)


def _piece(text: str, n: int, from_end: bool) -> str:
    return text[len(text) - n :] if from_end else text[:n]


def count_message(message: Message, counter: Counter) -> int:
    return counter.per_message + sum(_count_part(p, counter) for p in message.parts)


def count_messages(messages: Iterable[Message], counter: Counter) -> int:
    return sum(count_message(m, counter) for m in messages)


@dataclass(frozen=True)
class Budget:
    """`limit` is the model's input window as the caller decides it; `reserve_output` the room
    kept for the reply."""

    limit: int
    reserve_output: int

    def __post_init__(self) -> None:
        if self.reserve_output < 0 or self.reserve_output >= self.limit:
            raise ValueError(
                f"reserve_output must be in [0, limit): {self.reserve_output} of {self.limit}"
            )

    def available(self, fixed: Sequence[str | int], counter: Counter) -> int:
        """Tokens left for messages after the output reserve and the fixed content (system
        prompt, tool definitions: text, or a token count). Fixed content is never cut, so if it
        alone is over, nothing can help: `DoesNotFit`."""
        for f in fixed:
            if isinstance(f, bool) or (isinstance(f, int) and f < 0):
                raise ValueError(f"fixed token counts must be non-negative ints, got {f!r}")
        fixed_tokens = sum(f if isinstance(f, int) else counter.text(f) for f in fixed)
        left = self.limit - self.reserve_output - fixed_tokens
        if left < 0:
            report = Report(self.limit, self.reserve_output, fixed_tokens)
            raise DoesNotFit(
                f"fixed content is {fixed_tokens} tokens; only "
                f"{self.limit - self.reserve_output} fit beside the output reserve",
                report,
            )
        return left
