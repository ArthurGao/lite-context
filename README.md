# lite-context

**Not released yet** (S0, 2026-11-26: design draft; stage plan: design §14).

lite-context decides **what the model should see in this call**. Give it your full history — or a set of
documents or retrieved chunks — and a token budget; it returns what to send this time and a report of what it
truncated, masked or summarised, with token counts before and after. You still make the call through your own
LLM layer.

- **Your history is never rewritten.** A summary is a `Compaction` record that says which messages it covers,
  kept with the cut points in a `ContextState`;
  you store that next to your history and pass it back next time.
- **Nothing is dropped silently.** Only the steps you list may cut anything. If they are not enough, `fit()`
  raises `DoesNotFit` with the report.
- **Any message format.** Anthropic and OpenAI dicts are built in, lite-model is an extra, and your own types
  need a small adapter.
- **No model calls inside.** Summaries and map-reduce merges call a function you pass in, so the model call
  stays in your own role layer. The core is pure and has no dependencies, so it also runs inside a Temporal
  workflow.
- **Long documents:** split with overlap, pack pages into budget-sized groups, halve a batch a model refuses
  as too large, reduce notes hierarchically, pick retrieved chunks within a budget.
- **A REST API** (`litecontext serve`, loopback :8750) for hosts that are not Python.

## Planned usage

```python
from lite_context import Budget, fit
from lite_context.adapters import anthropic
from lite_context.steps import MaskToolResults, TruncateToolResults

messages = anthropic.to_messages(history)            # your stored history, untouched
fitted = fit(
    messages,
    budget=Budget(limit=200_000, reserve_output=8_000),
    fixed=[system_prompt],
    steps=[TruncateToolResults(max_tokens=2_000), MaskToolResults(keep_last=3)],
    state=saved_state,                               # or None; store fitted.state after the call
)
reply = my_llm.chat(system=system_prompt, messages=anthropic.from_messages(fitted.messages))
if fitted.report.compaction_due:
    ...                                              # write a summary with your own model call (design §6)
```

Design: [`docs/lite-context-design.md`](docs/lite-context-design.md).

## Developing

Python ≥ 3.11, uv. Guides for coding agents: [`AGENTS.md`](AGENTS.md).

## License

Apache-2.0.
