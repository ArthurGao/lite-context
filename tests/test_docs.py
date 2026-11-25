"""Long documents and retrieved chunks (design §7): split, pack, halving, reduce_tree, select.
Behaviour follows what the first hosts already do and the KB map-reduce guide §3."""

import asyncio
import itertools
import re
from collections.abc import Sequence

import pytest

from lite_context import DoesNotFit
from lite_context.docs import Piece, halving, pack, reduce_tree, select, split
from lite_context.testing import COUNTER  # one character is one token

# --- split ---


def test_text_that_fits_is_one_piece_exactly_as_given() -> None:
    assert split("short text", max_tokens=100, counter=COUNTER) == [
        Piece(0, 10, "short text", None)
    ]


def test_pieces_cover_the_text_overlap_and_end_at_a_line_break() -> None:
    lines = [f"line {i:03d} " + "x" * 38 for i in range(40)]  # 48 chars a line: 500 falls mid-line
    text = "\n".join(lines)
    pieces = split(text, max_tokens=500, counter=COUNTER, overlap=0.1)
    assert all(COUNTER.text(p.text) <= 500 for p in pieces)
    assert pieces[0].start == 0 and pieces[-1].end == len(text)
    for a, b in itertools.pairwise(pieces):
        assert b.start < a.end  # overlap
        assert a.end - b.start <= 0.1 * 500 + 1
        assert text[a.end - 1] == "\n" or text[a.end : a.end + 1] == "\n"  # cut at a line break
        assert p_text(text, a) == a.text


def p_text(text: str, p: Piece) -> str:
    return text[p.start : p.end]


def test_a_long_line_without_breaks_is_still_cut_to_size() -> None:
    pieces = split("y" * 1000, max_tokens=300, counter=COUNTER, overlap=0)
    assert [len(p.text) for p in pieces] == [300, 300, 300, 100]


def test_each_piece_carries_the_heading_of_the_section_it_starts_in() -> None:
    text = "SECTION A\n" + "a\n" * 200 + "SECTION B\n" + "b\n" * 200
    pieces = split(text, max_tokens=150, counter=COUNTER, heading=re.compile(r"^SECTION \w$"))
    assert pieces[0].heading == "SECTION A"  # it starts on that heading line
    assert {p.heading for p in pieces} == {"SECTION A", "SECTION B"}
    later = [p for p in pieces if p.start > text.index("SECTION B")]
    assert all(p.heading == "SECTION B" for p in later)


def test_a_bad_overlap_is_refused() -> None:
    with pytest.raises(ValueError, match="overlap"):
        split("x" * 100, max_tokens=10, counter=COUNTER, overlap=0.6)


# --- pack ---


def test_pack_groups_consecutive_units_under_the_budget_in_order() -> None:
    pages = ["a" * 40, "b" * 40, "c" * 40, "d" * 70, "e" * 10]
    groups = pack(pages, budget=100, cost=len, overhead=10)
    assert groups == [["a" * 40, "b" * 40], ["c" * 40], ["d" * 70, "e" * 10]]


def test_a_unit_over_the_budget_on_its_own_does_not_fit() -> None:
    with pytest.raises(DoesNotFit, match="unit 1"):
        pack(["a" * 10, "b" * 200], budget=100, cost=len)


# --- halving ---


class TooLarge(Exception):
    pass


class Spent(Exception):
    pass


def test_halving_halves_only_what_is_too_large_and_keeps_the_order() -> None:
    seen: list[tuple[int, ...]] = []

    async def call(items: Sequence[int]) -> list[int]:
        seen.append(tuple(items))
        if len(items) > 2:
            raise TooLarge
        return [i * 10 for i in items]

    done = asyncio.run(halving(list(range(5)), call, too_large=lambda e: isinstance(e, TooLarge)))
    assert done.results == [0, 10, 20, 30, 40]
    assert [(c.start, c.end) for c in done.calls] == [(0, 2), (2, 3), (3, 5)]
    assert seen == [(0, 1, 2, 3, 4), (0, 1), (2, 3, 4), (2,), (3, 4)]


def test_a_failure_that_is_not_about_size_is_raised_unchanged_after_one_call() -> None:
    calls: list[Sequence[int]] = []

    async def call(items: Sequence[int]) -> list[int]:
        calls.append(items)
        raise Spent

    with pytest.raises(Spent):
        asyncio.run(halving([1, 2, 3, 4], call, too_large=lambda e: isinstance(e, TooLarge)))
    assert calls == [[1, 2, 3, 4]]


def test_halving_stops_at_the_minimum_and_raises() -> None:
    async def call(items: Sequence[int]) -> list[int]:
        raise TooLarge

    with pytest.raises(TooLarge):
        asyncio.run(
            halving([1, 2, 3, 4], call, too_large=lambda e: isinstance(e, TooLarge), min_size=2)
        )


# --- reduce_tree ---


def test_notes_that_fit_together_come_back_untouched_with_no_merge() -> None:
    async def merge(group: list[str]) -> str:
        raise AssertionError("no merge needed")

    done = asyncio.run(reduce_tree(["a" * 10, "b" * 10], budget=100, cost=len, merge=merge))
    assert (done.notes, done.levels) == (["a" * 10, "b" * 10], [])


def test_adjacent_notes_are_merged_level_by_level_in_order() -> None:
    calls: list[list[str]] = []

    async def merge(group: list[str]) -> str:
        calls.append(group)
        return "(" + "+".join(n[0] for n in group) + ")" * 1

    notes = [c * 30 for c in "abcdefgh"]  # eight notes of 30, budget 100: three fit a group
    done = asyncio.run(reduce_tree(notes, budget=100, cost=len, merge=merge))
    assert calls[0] == [notes[0], notes[1], notes[2]]
    assert done.levels == [3]  # one level: 8 notes -> 3 groups, which then fit together
    assert done.notes == ["(a+b+c)", "(d+e+f)", "(g+h)"]


def test_a_note_too_long_to_merge_with_anything_fails_instead_of_looping() -> None:
    async def merge(group: list[str]) -> str:
        return "".join(group)

    with pytest.raises(DoesNotFit, match="too long"):
        asyncio.run(reduce_tree(["a" * 90, "b" * 90], budget=100, cost=len, merge=merge))


def test_a_merge_that_does_not_shrink_fails_at_the_depth_limit() -> None:
    async def merge(group: list[str]) -> str:
        return "z" * 60  # every merge is over half the budget: never fewer groups than before

    with pytest.raises(DoesNotFit):
        asyncio.run(reduce_tree(["a" * 40] * 6, budget=100, cost=len, merge=merge, max_depth=3))


def test_merging_stops_at_the_depth_limit_even_while_it_makes_progress() -> None:
    async def merge(group: list[str]) -> str:
        return "z" * 49  # 8 notes -> 4 (196 in all) -> 2 (98): two levels

    notes = ["a" * 40] * 8
    with pytest.raises(DoesNotFit, match="levels"):
        asyncio.run(reduce_tree(notes, budget=100, cost=len, merge=merge, max_depth=1))
    done = asyncio.run(reduce_tree(notes, budget=100, cost=len, merge=merge, max_depth=2))
    assert done.levels == [4, 2]


def test_a_merged_note_over_the_budget_fails() -> None:
    async def merge(group: list[str]) -> str:
        return "z" * 150

    with pytest.raises(DoesNotFit, match="merge"):
        asyncio.run(reduce_tree(["a" * 40] * 6, budget=100, cost=len, merge=merge))


# --- select ---


def test_select_takes_ranked_chunks_until_the_budget_and_drops_duplicates() -> None:
    chunks = [
        ("d1", "a" * 30),
        ("d2", "b" * 30),
        ("d1", "a" * 30),
        ("d3", "c" * 50),
        ("d4", "d" * 5),
    ]
    done = select(chunks, budget=70, cost=lambda c: len(c[1]), key=lambda c: c[0])
    assert done.chosen == [chunks[0], chunks[1]]
    assert done.duplicates == [chunks[2]]
    assert done.over_budget == [chunks[3], chunks[4]]  # stops at the first that does not fit


def test_split_respects_the_limit_with_the_default_estimate_and_cuts_a_large_text_quickly() -> None:
    import time

    from lite_context import CharEstimate

    text = "\n".join(f"paragraph {i}: " + "word " * (i % 37) for i in range(40_000))  # ~1 MB
    t0 = time.perf_counter()
    pieces = split(text, max_tokens=8_000, counter=CharEstimate())
    elapsed = time.perf_counter() - t0
    assert all(CharEstimate().text(p.text) <= 8_000 for p in pieces)
    assert pieces[-1].end == len(text)
    assert all(b.start < a.end for a, b in itertools.pairwise(pieces))
    assert elapsed < 1.0, elapsed


# --- regressions from the C2 review ---


def test_halving_needs_a_minimum_of_at_least_one() -> None:
    async def call(items: Sequence[int]) -> list[int]:
        raise TooLarge

    with pytest.raises(ValueError, match="min_size"):
        asyncio.run(halving([1], call, too_large=lambda e: True, min_size=0))


class Stop(BaseException):
    pass


def test_a_cancellation_never_reaches_the_size_predicate() -> None:
    asked: list[BaseException] = []

    async def call(items: Sequence[int]) -> list[int]:
        raise Stop

    def too_large(e: Exception) -> bool:
        asked.append(e)
        return True

    with pytest.raises(Stop):
        asyncio.run(halving([1, 2], call, too_large=too_large))
    assert asked == []


def test_halving_says_which_results_each_call_gave() -> None:
    async def call(items: Sequence[int]) -> list[str]:
        if len(items) > 1:
            raise TooLarge
        return [f"n{items[0]}", f"m{items[0]}"]

    done = asyncio.run(halving([1, 2], call, too_large=lambda e: isinstance(e, TooLarge)))
    assert [(c.start, c.end, c.results) for c in done.calls] == [
        (0, 1, ["n1", "m1"]),
        (1, 2, ["n2", "m2"]),
    ]
    assert done.results == ["n1", "m1", "n2", "m2"]


def test_select_sorts_every_chunk_into_exactly_one_list() -> None:
    k1, k2, k1b, k3, k3b = ("k1", 5), ("k2", 10), ("k1", 5), ("k3", 1), ("k3", 1)
    done = select([k1, k2, k1b, k3, k3b], budget=6, cost=lambda c: c[1], key=lambda c: c[0])
    assert done.chosen == [k1]
    assert done.duplicates == [k1b, k3b]
    assert done.over_budget == [k2, k3]


def test_split_counts_a_bounded_window_not_the_whole_rest_each_time() -> None:
    from lite_context import CharEstimate

    class Tally(CharEstimate):
        counted = 0

        def text(self, s: str) -> int:
            Tally.counted += len(s)
            return super().text(s)

    text = "\n".join("w" * 60 for _ in range(16_000))  # ~1 MB
    pieces = split(text, max_tokens=3_000, counter=Tally())
    assert len(pieces) > 50
    assert Tally.counted < 12 * len(text), Tally.counted / len(text)


def test_a_piece_that_starts_on_a_heading_carries_that_heading() -> None:
    text = "A\n" + "a" * 7 + "\nB\n" + "b" * 7 + "\n"
    pieces = split(text, max_tokens=10, counter=COUNTER, overlap=0, heading=re.compile(r"^[AB]$"))
    starts_at_b = [p for p in pieces if p.text.startswith("B\n")]
    assert starts_at_b and starts_at_b[0].heading == "B"


def test_a_counter_that_charges_more_for_the_whole_than_its_parts_is_still_respected() -> None:
    from lite_context import CharEstimate

    class Superadditive(CharEstimate):
        def text(self, s: str) -> int:
            return len(s) + len(s) ** 2 // 2000  # longer strings cost more per character

    lines = "\n".join("v" * 49 for _ in range(200))
    counter = Superadditive()
    pieces = split(lines, max_tokens=600, counter=counter, overlap=0)
    assert all(counter.text(p.text) <= 600 for p in pieces)
    assert pieces[-1].end == len(lines)
