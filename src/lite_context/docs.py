"""Long documents and retrieved chunks (design §7). Model calls are the caller's functions.

The rules come from the KB map-reduce guide §3 and the first hosts: count first, and when the
text fits, give it back whole so the call is the one an unsplit text would make; group adjacent
notes by budget and merge them level by level; a note too long to merge with anything, or a merge
that does not get under the budget, fails the whole reduce, never a fallback to concatenation;
halve only what the caller says was refused as too large, and raise anything else unchanged.
"""

from __future__ import annotations

import bisect
import re
from collections.abc import Awaitable, Callable, Hashable, Sequence
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from .count import CharEstimate, Counter, longest_prefix
from .errors import DoesNotFit
from .report import Report

T = TypeVar("T")
R = TypeVar("R")


# --- split ---


@dataclass(frozen=True)
class Piece:
    start: int
    end: int
    text: str
    # The heading of the section the piece starts in: context for a piece that begins mid-section.
    heading: str | None


def split(
    text: str,
    *,
    max_tokens: int,
    counter: Counter | None = None,
    overlap: float = 0.06,
    prefer: str = "\n",
    heading: re.Pattern[str] | None = None,
) -> list[Piece]:
    """Pieces of at most `max_tokens` covering `text`. Text that fits is one piece, as given.
    Otherwise the text is broken after each `prefer` into segments (a segment over the limit is
    cut to size), each counted once; a piece is as many whole segments as fit, checked with one
    real count; the next starts about `overlap` of its length back, on a segment boundary. Summing
    segment counts errs high for subword tokenizers, the safe direction (KB map-reduce guide §3).
    Empty text gives no pieces."""
    counter = counter or CharEstimate()
    if not 0 <= overlap < 0.5:
        raise ValueError(f"overlap must be in [0, 0.5), got {overlap}")
    if max_tokens < 1:
        raise _too_big(f"max_tokens is {max_tokens}", max_tokens, 0)
    if counter.text(text) <= max_tokens:
        return [Piece(0, len(text), text, None)] if text else []
    segments = _segments(text, prefer, max_tokens, counter)  # (start, end, tokens)
    headings = _headings(text, heading)
    pieces: list[Piece] = []
    i = 0
    while i < len(segments):
        j, used = i, 0
        while j < len(segments) and used + segments[j][2] <= max_tokens:
            used += segments[j][2]
            j += 1
        if j == i:  # only a counter that disagrees with itself gets here; fail, never loop
            raise _too_big(f"segment {i} no longer fits {max_tokens} tokens", max_tokens, 0)
        while j - i > 1 and counter.text(text[segments[i][0] : segments[j - 1][1]]) > max_tokens:
            j -= 1  # a counter that is not additive
        start, end = segments[i][0], segments[j - 1][1]
        pieces.append(Piece(start, end, text[start:end], _section(headings, start)))
        if j == len(segments):
            break
        back = j
        while back - 1 > i and end - segments[back - 1][0] <= (end - start) * overlap:
            back -= 1
        i = back
    return pieces


def _segments(
    text: str, prefer: str, max_tokens: int, counter: Counter
) -> list[tuple[int, int, int]]:
    """Text broken just after each `prefer`; a segment over `max_tokens` cut to size."""
    out: list[tuple[int, int, int]] = []
    start = 0
    while start < len(text):
        at = text.find(prefer, start) if prefer else -1
        end = len(text) if at < 0 else at + len(prefer)
        while start < end:
            piece = text[start:end]
            tokens = counter.text(piece)
            if tokens > max_tokens:
                piece = longest_prefix(piece, max_tokens, counter)
                if not piece:
                    raise _too_big(f"not one character fits {max_tokens} tokens", max_tokens, 0)
                tokens = counter.text(piece)
            out.append((start, start + len(piece), tokens))
            start += len(piece)
    return out


def _headings(text: str, pattern: re.Pattern[str] | None) -> list[tuple[int, str]]:
    """(line start, heading) for each line matching `pattern`, in order."""
    if pattern is None:
        return []
    found: list[tuple[int, str]] = []
    at = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped and pattern.match(stripped):
            found.append((at, stripped))
        at += len(line)
    return found


def _section(headings: list[tuple[int, str]], start: int) -> str | None:
    """The last heading at or above `start`: a piece that starts on a heading line is in it."""
    i = bisect.bisect_right(headings, start, key=lambda h: h[0])
    return headings[i - 1][1] if i else None


def _too_big(why: str, budget: int, overhead: int) -> DoesNotFit:
    return DoesNotFit(why, Report(limit=budget, reserve_output=0, fixed_tokens=overhead))


# --- pack ---


def pack(
    units: Sequence[T], *, budget: int, cost: Callable[[T], int], overhead: int = 0
) -> list[list[T]]:
    """Consecutive units grouped so each group, with `overhead`, costs at most `budget`; order
    kept, every unit in exactly one group. A unit that cannot fit on its own: `DoesNotFit`."""
    groups: list[list[T]] = []
    current: list[T] = []
    used = overhead
    for i, unit in enumerate(units):
        size = cost(unit)
        if overhead + size > budget:
            raise _too_big(f"unit {i} costs {size}; {budget - overhead} fit", budget, overhead)
        if current and used + size > budget:
            groups.append(current)
            current, used = [], overhead
        current.append(unit)
        used += size
    if current:
        groups.append(current)
    return groups


# --- halving ---


@dataclass(frozen=True)
class Call(Generic[R]):
    start: int  # the items[start:end] one successful call read
    end: int
    results: list[R]


@dataclass(frozen=True)
class Halved(Generic[R]):
    calls: list[Call[R]]  # in item order

    @property
    def results(self) -> list[R]:
        return [r for c in self.calls for r in c.results]


async def halving(
    items: Sequence[T],
    call: Callable[[Sequence[T]], Awaitable[list[R]]],
    *,
    too_large: Callable[[Exception], bool],
    min_size: int = 1,
) -> Halved[R]:
    """`call(items)`; when it raises what `too_large` says is a size refusal, the two halves are
    read one after the other, in order. Any other failure, a refusal of `min_size` items or fewer,
    and a cancellation (never shown to `too_large`) are raised unchanged."""
    if min_size < 1:
        raise ValueError(f"min_size must be at least 1, got {min_size}")
    calls: list[Call[R]] = []

    async def read(lo: int, hi: int) -> None:
        try:
            out = await call(items[lo:hi])
        except Exception as exc:
            if hi - lo <= min_size or not too_large(exc):
                raise
            mid = lo + (hi - lo) // 2
            await read(lo, mid)
            await read(mid, hi)
            return
        calls.append(Call(lo, hi, list(out)))

    await read(0, len(items))
    return Halved(calls)


# --- reduce ---


@dataclass(frozen=True)
class Reduced(Generic[T]):
    notes: list[T]  # these fit the budget together: the caller's final call reads them
    levels: list[int] = field(default_factory=list)  # how many groups each merge level made


async def reduce_tree(
    notes: Sequence[T],
    *,
    budget: int,
    cost: Callable[[T], int],
    merge: Callable[[list[T]], Awaitable[T]],
    overhead: int = 0,
    max_depth: int = 8,
) -> Reduced[T]:
    """Merge adjacent notes, a level at a time, until they fit `budget` together. Groups are
    merged one after another (on a capped account the per-minute allowance is the bound, and
    concurrent merges are refused for it). Fails rather than loops or concatenates."""
    level = [(n, cost(n)) for n in notes]
    levels: list[int] = []
    while overhead + sum(size for _, size in level) > budget:
        if len(levels) == max_depth:
            raise _too_big(
                f"still {len(level)} notes over the budget after {max_depth} levels",
                budget,
                overhead,
            )
        groups = pack(level, budget=budget, cost=lambda pair: pair[1], overhead=overhead)
        if len(groups) == len(level):
            raise _too_big(
                f"each of the {len(level)} notes is too long to merge with another in {budget}",
                budget,
                overhead,
            )
        merged: list[tuple[T, int]] = []
        for group in groups:
            if len(group) == 1:
                merged.append(group[0])
                continue
            note = await merge([n for n, _ in group])
            size = cost(note)
            if overhead + size > budget:
                raise _too_big(
                    f"a merge returned {size} tokens; {budget - overhead} fit", budget, overhead
                )
            merged.append((note, size))
        level = merged
        levels.append(len(groups))
    return Reduced([n for n, _ in level], levels)


# --- select ---


@dataclass(frozen=True)
class Selected(Generic[T]):
    chosen: list[T]
    duplicates: list[T]
    over_budget: list[T]  # from the first chunk that did not fit on


def select(
    chunks: Sequence[T], *, budget: int, cost: Callable[[T], int], key: Callable[[T], Hashable]
) -> Selected[T]:
    """Chunks already ranked by the caller, taken in that order while they fit; a chunk whose
    key was taken already is a duplicate. Stops at the first that does not fit, so a lower-ranked
    short chunk never displaces a higher-ranked one."""
    chosen: list[T] = []
    duplicates: list[T] = []
    over_budget: list[T] = []
    seen: set[Hashable] = set()
    used = 0
    for chunk in chunks:
        k = key(chunk)
        if k in seen:
            duplicates.append(chunk)
            continue
        seen.add(k)
        size = cost(chunk)
        if over_budget or used + size > budget:
            over_budget.append(chunk)
        else:
            chosen.append(chunk)
            used += size
    return Selected(chosen, duplicates, over_budget)


__all__ = [
    "Call",
    "Halved",
    "Piece",
    "Reduced",
    "Selected",
    "halving",
    "pack",
    "reduce_tree",
    "select",
    "split",
]
