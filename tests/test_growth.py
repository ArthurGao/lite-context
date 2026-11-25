"""Random histories that grow one message per call, the state passed back (design §5.4): no
error but `DoesNotFit`, the same output for the same input, and, while the state does not move,
the previous request stays a prefix of the next one, except a message that has just left
protection (design §5.4 accepts that one change)."""

import random

import pytest

from lite_context import (
    Budget,
    ContextState,
    DoesNotFit,
    Message,
    Opaque,
    Text,
    ToolCall,
    ToolResult,
    fit,
)
from lite_context.count import count_messages
from lite_context.steps import DropOldest, MaskOpaque, MaskToolResults, Step, TruncateToolResults
from lite_context.structure import last_human
from lite_context.testing import COUNTER


def history(r: random.Random) -> list[Message]:
    out: list[Message] = []
    for t in range(r.randint(6, 16)):
        parts: list[Text | Opaque] = [Text("q" * r.randint(1, 30))]
        if r.random() < 0.3:
            parts.append(Opaque(kind="image", raw={"t": t}))
        out.append(Message("user", tuple(parts), key=f"k{len(out)}", pinned=r.random() < 0.08))
        for b in range(r.randint(0, 3)):
            ids = [f"c{t}_{b}_{j}" for j in range(r.randint(1, 3))]
            calls = tuple(ToolCall(i, "s", "{}") for i in ids)
            thinking = Opaque(kind="thinking", raw={"x": 1}, text="th" * 5)
            out.append(Message("assistant", (thinking, *calls), key=f"k{len(out)}"))
            groups = [ids] if r.random() < 0.7 else [[i] for i in ids]
            for g in groups:
                res = tuple(
                    ToolResult(
                        i,
                        (
                            Text("x" * r.randint(5, 400)),
                            *(
                                Opaque(kind="image", raw={"i": i, "j": j})
                                for j in range(r.choice([0, 0, 1, 2]))
                            ),
                        ),
                    )
                    for i in g
                )
                out.append(Message("user", res, key=f"k{len(out)}"))
        out.append(Message("assistant", (Text("a" * r.randint(1, 50)),), key=f"k{len(out)}"))
    return out


@pytest.mark.parametrize("seed", range(150))
def test_a_growing_history_replays_stably(seed: int) -> None:
    r = random.Random(seed)
    msgs = history(r)
    pool: list[Step] = [
        MaskToolResults(r.randint(0, 3)),
        MaskOpaque(r.randint(0, 3)),
        DropOldest(r.randint(1, 4)),
    ]
    steps = r.sample(pool, r.randint(1, 3))
    if r.random() < 0.5:
        steps = [TruncateToolResults(r.randint(20, 200)), *steps]
    budget = Budget(int(count_messages(msgs, COUNTER) * r.uniform(0.3, 0.9)), 0)
    state, previous_human = ContextState(), None
    previous: list[Message] = []
    for end in range(1, len(msgs) + 1):
        try:
            one = fit(msgs[:end], budget=budget, steps=steps, state=state, counter=COUNTER)
        except DoesNotFit:
            return
        two = fit(msgs[:end], budget=budget, steps=steps, state=state, counter=COUNTER)
        assert (one.messages, one.state) == (two.messages, two.state)
        if one.state == state:
            changed = [a.key for a, b in zip(previous, one.messages, strict=False) if a != b]
            assert set(changed) <= {previous_human}, (end, changed)
        h = last_human(msgs[:end])
        state, previous, previous_human = (
            one.state,
            one.messages,
            msgs[h].key if h is not None else None,
        )
