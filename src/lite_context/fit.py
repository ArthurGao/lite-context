"""`fit()`: what to send this call (design §5)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from .count import Budget, CharEstimate, Counter, count_message
from .errors import DoesNotFit, ProtectedChange, StaleState, UnreportedChange
from .report import Action, Report
from .state import ContextState, Cut, digest
from .steps import CuttingStep, Entry, Step, StepContext, Unit
from .structure import batches, last_human
from .types import Message, Opaque, ToolResult, check_unique


@dataclass(frozen=True)
class Fitted:
    messages: list[Message]
    state: ContextState  # store it next to the history and pass it back next call
    report: Report


def fit(
    messages: Sequence[Message],
    *,
    budget: Budget,
    steps: Sequence[Step] = (),
    fixed: Sequence[str | int] = (),
    state: ContextState | None = None,
    counter: Counter | None = None,
    high: float = 0.9,
    low: float = 0.7,
) -> Fitted:
    """Fit `messages` into `budget`. Always steps run on every call; cutting steps replay the cut
    in `state`, and cut further only while over the high water mark, down to the low one. Raises
    `DoesNotFit` when the steps are not enough; the input is never changed."""
    counter = counter or CharEstimate()
    if not 0 < low < high <= 1:
        raise ValueError(f"water marks must satisfy 0 < low < high <= 1, got {low}, {high}")
    names = [s.name for s in steps]
    if dup := {n for n in names if names.count(n) > 1}:
        raise ValueError(f"two steps named {sorted(dup)}")
    available = budget.available(fixed, counter)
    entries = _entries(messages)
    keyed = all(m.key is not None for m in messages)
    state = state or ContextState()
    if state.cuts and not keyed:
        raise StaleState("the state holds cuts, but the messages have no keys")
    batches(messages)

    ctx = StepContext(counter, _protected(entries))
    order = {
        e.ref: i for i, e in enumerate(entries)
    }  # positions in the input: cuts are measured here
    input_refs = [e.ref for e in entries]
    high_mark, target = math.floor(available * high), math.floor(available * low)
    tally = _Tally(entries, counter)
    tokens_before = tally.total
    actions: list[Action] = []
    cuts: dict[str, Cut] = {}
    for step in steps:
        if step.always:
            after, done = step.apply(entries, ctx)
            _check(entries, after, done, ctx)
        else:
            through = _replayed(step.name, state, order, input_refs)
            # Units are put in input order here, whatever order the step listed them in.
            placed = sorted(
                (
                    (p, u)
                    for u in step.units(entries, ctx)
                    if (p := _position(u, order)) is not None
                ),
                key=lambda pu: pu[0],
            )
            spots, units = [p for p, _ in placed], [u for _, u in placed]
            n = sum(1 for p in spots if p <= through)
            after, done = _cut(step, entries, units, n, ctx)
            # Hysteresis: cut further only when over the high mark, and then down to the low one;
            # never take a cut that costs more than it saves (a placeholder can).
            replayed = tally.after(after, done)
            if replayed > high_mark and n < len(units):
                m, more, more_done = _least(step, entries, units, spots, n, target, tally, ctx)
                if tally.after(more, more_done) < replayed:
                    n, after, done = m, more, more_done
            if n and keyed:
                last = spots[n - 1]  # units are in input order
                cuts[step.name] = Cut(input_refs[last], digest(input_refs[: last + 1]))
        tally.update(after, done)
        entries = after
        actions += done

    tokens_after = tally.total
    report = Report(
        limit=budget.limit,
        reserve_output=budget.reserve_output,
        fixed_tokens=budget.limit - budget.reserve_output - available,
        available=available,
        tokens_before=tokens_before,
        tokens_after=tokens_after,
        actions=tuple(actions),
        compaction_due=tokens_after > high_mark,
        estimated=tuple(e.ref for e in entries if _estimated(e.message)),
    )
    if tokens_after > available:
        raise DoesNotFit(f"{tokens_after} tokens after every step; {available} available", report)
    return Fitted([e.message for e in entries], ContextState(cuts), report)


def _entries(messages: Sequence[Message]) -> list[Entry]:
    keys = [m.key for m in messages if m.key is not None]
    if reserved := [k for k in keys if k.startswith(("#", "+"))]:
        raise ValueError(f"keys starting with '#' or '+' are reserved for lite-context: {reserved}")
    check_unique(keys)
    return [Entry(m.key if m.key is not None else f"#{i}", m) for i, m in enumerate(messages)]


def _protected(entries: Sequence[Entry]) -> frozenset[str]:
    refs = {e.ref for e in entries if e.message.pinned}
    human = last_human([e.message for e in entries])
    if human is not None:
        refs.add(entries[human].ref)
    return frozenset(refs)


def _position(unit: Unit, order: dict[str, int]) -> int | None:
    """A unit's place in the input: its newest message. None for a unit over messages a step
    inserted (a placeholder), which is never cut."""
    if not unit.refs or any(r not in order for r in unit.refs):
        return None
    return max(order[r] for r in unit.refs)


def _replayed(name: str, state: ContextState, order: dict[str, int], refs: Sequence[str]) -> int:
    """The input position the step's earlier cut reached, or -1. Units are cut by position, so a
    unit that another step already removed, or that left protection, does not shift the cut."""
    cut = state.cuts.get(name)
    if cut is None:
        return -1
    at = order.get(cut.through)
    if at is None or digest(refs[: at + 1]) != cut.digest:
        raise StaleState(f"{name}: the history changed before its cut")
    return at


class _Tally:
    """Token counts per ref, counted once; after a step only the messages it reported are
    counted again (`_check` proves the others unchanged)."""

    def __init__(self, entries: Sequence[Entry], counter: Counter) -> None:
        self.counter = counter
        self.tokens = {e.ref: count_message(e.message, counter) for e in entries}
        self.total = sum(self.tokens.values())

    def after(self, entries: Sequence[Entry], actions: Sequence[Action]) -> int:
        touched = {a.ref for a in actions}
        now = {e.ref: e.message for e in entries if e.ref in touched}
        gone = sum(self.tokens.get(r, 0) for r in touched)
        return self.total - gone + sum(count_message(m, self.counter) for m in now.values())

    def deltas(self, entries: Sequence[Entry], actions: Sequence[Action]) -> dict[str, int]:
        """Tokens saved per reported ref (negative for an inserted message)."""
        now = {e.ref: e.message for e in entries}
        return {
            r: self.tokens.get(r, 0) - (count_message(now[r], self.counter) if r in now else 0)
            for r in {a.ref for a in actions}
        }

    def update(self, entries: Sequence[Entry], actions: Sequence[Action]) -> None:
        self.total = self.after(entries, actions)
        for ref in {a.ref for a in actions}:
            self.tokens.pop(ref, None)
        for e in entries:
            if e.ref not in self.tokens:
                self.tokens[e.ref] = count_message(e.message, self.counter)


def _cut(
    step: CuttingStep, entries: Sequence[Entry], units: Sequence[Unit], n: int, ctx: StepContext
) -> tuple[list[Entry], list[Action]]:
    """The step's output for its first `n` units, checked against design §5.2 whether or not
    `fit()` ends up using it."""
    if not n:
        return list(entries), []
    after, done = step.apply(entries, units[:n], ctx)
    _check(entries, after, done, ctx)
    return after, done


def _least(
    step: CuttingStep,
    entries: Sequence[Entry],
    units: Sequence[Unit],
    spots: Sequence[int],
    n: int,
    target: int,
    tally: _Tally,
    ctx: StepContext,
) -> tuple[int, list[Entry], list[Action]]:
    """The fewest units past `n` that bring the total to `target`, or else the cheapest count.

    Counts end only at a message boundary: a cut records a whole message (design §5.4), so units
    sharing a position go together. Cutting more does not always cost less (a placeholder can be
    longer than what it replaces), so nothing assumes it. Each unit's saving is read from one
    application of every unit; the totals of all counts follow by a running sum; the chosen count
    is then applied and counted exactly, stepping on while the estimate proved too low."""
    every, done_all = _cut(step, entries, units, len(units), ctx)
    owner = {r: k for k, u in enumerate(units) for r in u.refs}
    saving = [0] * len(units)
    extra = 0  # messages the step inserted belong to no unit: a cost paid once
    for ref, delta in tally.deltas(every, done_all).items():
        if ref in owner:
            saving[owner[ref]] += delta
        else:
            extra -= delta
    base = tally.total
    bounds = [
        m for m in range(n + 1, len(units) + 1) if m == len(units) or spots[m] != spots[m - 1]
    ]
    running, estimates = sum(saving[:n]), {}
    for m in range(n + 1, len(units) + 1):
        running += saving[m - 1]
        estimates[m] = base - running + extra
    start = next(
        (m for m in bounds if estimates[m] <= target), min(bounds, key=estimates.__getitem__)
    )
    best: tuple[int, list[Entry], list[Action]] | None = None
    best_total = 0
    for m in [b for b in bounds if b >= start] or [start]:
        after, done = _cut(step, entries, units, m, ctx)
        total = tally.after(after, done)
        if best is None or total < best_total:
            best, best_total = (m, after, done), total
        if total <= target or estimates[m] > target:
            break  # reached it, or no further count was expected to
    assert best is not None
    return best


def _estimated(m: Message) -> bool:
    for p in m.parts:
        inner = p.content if isinstance(p, ToolResult) else (p,)
        if any(isinstance(c, Opaque) and c.tokens is None for c in inner):
            return True
    return False


def _thinking(m: Message) -> list[Opaque]:
    return [p for p in m.parts if isinstance(p, Opaque) and p.kind == "thinking"]


def _check(
    before: Sequence[Entry], after: Sequence[Entry], actions: Sequence[Action], ctx: StepContext
) -> None:
    """Design §5.2 at run time, after every step (the host's own included)."""
    touched = {a.ref for a in actions}
    if len({e.ref for e in after}) != len(after):
        raise UnreportedChange("a step returned two messages with one ref")
    old = {e.ref: e.message for e in before}
    new = {e.ref: e.message for e in after}
    for ref in ctx.protected:
        if new.get(ref) != old[ref]:
            raise ProtectedChange(f"{ref}: a step changed a protected message")
    for ref in touched:
        if ref in old and ref in new and _thinking(new[ref]) != _thinking(old[ref]):
            raise ProtectedChange(f"{ref}: a step changed a thinking block")
    kept_before = [e.ref for e in before if e.ref not in touched]
    kept_after = [e.ref for e in after if e.ref not in touched]
    if kept_before != kept_after or any(new[r] != old[r] for r in kept_after):
        raise UnreportedChange("a step changed, moved or removed a message it did not report")
    if any(e.ref not in old and e.ref not in touched for e in after):
        raise UnreportedChange("a step added a message it did not report")
    batches([e.message for e in after])
