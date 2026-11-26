# lite-context

**v0.2.0** (2026-11-26; stage plan: design §14).

lite-context decides **what the model should see in this call**. Give it your full history (or a set of
documents or retrieved chunks) and a token budget. It returns what to send this time and a report of what it
truncated, masked or summarised, with token counts before and after. You still make the call through your own
LLM layer.

- **Your history is never rewritten.** A summary is a `Compaction` record that says which messages it covers.
  It is kept with the cut points in a `ContextState`, which you store next to your history and pass back next time.
- **The prompt cache keeps working.** Old messages change only on the calls where a cut moves forward. While it
  does not move, each request starts with the previous one, unchanged.
- **Nothing is dropped silently.** Only the steps you list may cut anything. If they are not enough, `fit()`
  raises `DoesNotFit` with the report. Every step's output is checked at run time: tool calls keep their results,
  and pinned messages, the latest human turn and thinking blocks stay untouched.
- **Any message format.** Anthropic and OpenAI dicts are built in and round-trip losslessly. Your own types need
  a small adapter.
- **No model calls inside.** Summaries and map-reduce merges call a function you pass in, so the model call stays
  in your own role layer. The core is pure and has no dependencies, so it also runs inside a Temporal workflow.
- **No storage of your own?** `Transcript` keeps an append-only log of messages and states, in memory or in
  PostgreSQL (extra `pg`), and projects it through `fit()`.
- **Long documents:** split with overlap and section headings, pack pages into budget-sized groups, halve a batch
  a model refuses as too large, merge notes level by level, pick retrieved chunks within a budget.

## Install

From the GitHub Release (lite-context is not on PyPI). Python 3.11 or 3.12, no dependencies:

```
uv add "lite-context @ https://github.com/ArthurGao/lite-context/releases/download/v0.2.0/lite_context-0.2.0-py3-none-any.whl"
pip install "lite-context @ https://github.com/ArthurGao/lite-context/releases/download/v0.2.0/lite_context-0.2.0-py3-none-any.whl"
pip install "lite-context[pg] @ https://github.com/ArthurGao/lite-context/releases/download/v0.2.0/lite_context-0.2.0-py3-none-any.whl"   # PgStore
```

## Example

```python
from lite_context import Budget, ContextState, fit, plan_compaction, with_keys
from lite_context.adapters import anthropic
from lite_context.steps import MaskToolResults, TruncateToolResults

messages = with_keys(anthropic.to_messages(history), row_ids)   # your stored history, untouched
state = ContextState.from_json(saved) if saved else None

fitted = fit(
    messages,
    budget=Budget(limit=200_000, reserve_output=8_000),
    fixed=[system_prompt],
    steps=[TruncateToolResults(max_tokens=2_000), MaskToolResults(keep_last=3)],
    state=state,
)
reply = my_llm.chat(system=system_prompt, messages=anthropic.from_messages(fitted.messages))
saved = fitted.state.to_json()                                   # store it next to the history

if fitted.report.compaction_due:                                 # still over 90% after cutting
    plan = plan_compaction(messages, state=fitted.state)
    if plan is not None:
        summary = my_llm.chat(role="compact", system=plan.system,
                              messages=anthropic.from_messages(plan.prompt))
        saved = plan.accept(summary.text, truncated=summary.hit_limit).to_json()
```

If you build the step list in a variable first, annotate it (`steps: list[Step] = [...]`), or mypy infers
`list[object]`.

Long documents:

```python
from lite_context.docs import halving, pack, reduce_tree, select, split

pieces = split(text, max_tokens=80_000, heading=re.compile(r"^(SECTION|PART)\b.*"))
groups = pack(pages, budget=100_000, cost=lambda p: estimate(p.text))
notes = await halving(lines, read_part, too_large=lambda e: getattr(e, "oversized", False))
merged = await reduce_tree(notes.results, budget=24_000, cost=estimate, merge=merge_notes)
```

A transcript, for a host with no history store (PostgreSQL: `litecontext migrate` once, URL in
`LITECONTEXT_DATABASE_URL`):

```python
from lite_context import Budget, InMemoryStore, Transcript
from lite_context.pg import PgStore            # extra `pg`

t = Transcript(PgStore(url), "thread-1")       # or InMemoryStore()
await t.create()
seq = await t.append([question], expected_seq=await t.last_seq())   # SeqConflict if someone was first
fitted = await t.project(budget=Budget(200_000, 8_000), steps=steps)
reply = my_llm.chat(messages=anthropic.from_messages(fitted.messages))
await t.append([answer, fitted.state], expected_seq=seq)            # the state is an entry too
```

`scripts/smoke.py` runs a complete example against the installed wheel.

Design: [`docs/lite-context-design.md`](docs/lite-context-design.md).

## Developing

Python ≥ 3.11, uv. Guides for coding agents: [`AGENTS.md`](AGENTS.md).

## License

Apache-2.0.
