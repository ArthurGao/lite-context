"""The installed wheel works from a clean environment (CI `package` job, design §14 C3): the
README's example runs and gives the expected result. Run with `python -I` from outside the repo,
so the installed package is the one imported."""

import asyncio
import json
from importlib.metadata import version

from lite_context import Budget, ContextState, Message, Summary, acompact, fit, with_keys
from lite_context.adapters import anthropic
from lite_context.docs import split
from lite_context.steps import MaskToolResults, Step, TruncateToolResults


def history(turns: int) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for t in range(turns):
        out += [
            {"role": "user", "content": f"question {t}"},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": f"c{t}", "name": "search", "input": {"q": t}}
                ],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": f"c{t}", "content": "x" * 4000}],
            },
            {"role": "assistant", "content": f"answer {t}"},
        ]
    return out


async def main() -> None:
    native = history(12)
    messages = with_keys(anthropic.to_messages(native), [f"m{i}" for i in range(len(native))])
    steps: list[Step] = [TruncateToolResults(max_tokens=500), MaskToolResults(keep_last=2)]
    fitted = fit(messages, budget=Budget(6_000, 1_000), fixed=["be brief"], steps=steps)
    sent = anthropic.from_messages(fitted.messages)
    assert fitted.report.tokens_after <= fitted.report.available, fitted.report
    assert any(a.what == "masked" for a in fitted.report.actions)
    assert sent[0] == native[0]  # untouched messages go out as they came in

    state = ContextState.from_json(fitted.state.to_json())

    async def summarize(system: str, prompt: list[Message]) -> Summary:
        return Summary("they asked twelve questions", provenance={"role": "compact"})

    summarised = await acompact(messages, state=state, summarize=summarize)
    assert summarised is not None and summarised.compaction is not None
    again = fit(messages, budget=Budget(6_000, 1_000), steps=steps, state=summarised)
    assert again.messages[0].parts[0].text.endswith("they asked twelve questions")  # type: ignore[union-attr]

    pieces = split("line\n" * 10_000, max_tokens=1_000)
    assert len(pieces) > 1 and pieces[-1].end == 50_000
    print(json.dumps({"lite-context": version("lite-context"), "pieces": len(pieces)}), "ok")


if __name__ == "__main__":
    asyncio.run(main())
