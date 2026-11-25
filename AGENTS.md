# lite-context — Agent Guide

lite-context is a **context-management library** for LLM calls. It answers one question: *what should the
model see in this call?* Given the caller's full history (or a set of documents or retrieved chunks) and a
token budget, it returns what to send this time plus a report of what it cut, masked or summarised. The
caller still makes the call through its own LLM layer. It serves projects that call an LLM directly (with
their own message types) and lite-agent alike. It is a library first; the REST API is a thin shell over it.

This file is the shared guide for every coding agent. `CLAUDE.md` imports it.

Naming: repo `lite-context`, Python package `lite_context`, CLI `litecontext`, env prefix `LITECONTEXT_`.

## Sources of truth (highest first)

1. `docs/lite-context-design.md` — the design. Its first line is the project status; §0 the decisions (who made
   each one); §3 types and adapters; §5 `fit()` and its rules; §6 compaction; §7 documents; §14 the stages and
   their acceptance; §15 open questions.
2. Project KB: `/Users/arthurgao/Docs/Knowledgebase/projects/lite-context/` — owner decisions, measured results
   from host adoption, lessons. The rules this project puts into practice live in
   `KB/global/llm-app-architecture.md` §9 and §11; when a measurement contradicts one, fix that page.
   Write with the `kb-maintain` skill.

If a change must contradict the design, update the design in the same change and say why in the commit.

## Current stage

**S0 done 2026-11-26**: repository, design, agent guides, project KB. **C0 done 2026-11-26**: the neutral types,
the Anthropic and OpenAI adapters (lossless round trip), counting and budgets (design §3, §4). **C1a done
2026-11-26**: `fit()`, the four steps, `ContextState` with cuts by position, the run-time checks after every step,
`testing.check_step` (design §5). **C1b done 2026-11-26**: summaries — `plan_compaction`, `accept`, `acompact`,
`SummaryPolicy`, applied by `fit()` under the same checks (design §6). C1 is complete. **C2 done 2026-11-26**:
long documents — `split`, `pack`, `halving`, `reduce_tree`, `select` (design §7). **C3 prepared**: version 0.1.0,
CI `package` / `release` jobs, `scripts/smoke.py`; the release is the `v0.1.0` tag (owner's go-ahead).

## Rules

1. **Never mutate the caller's history.** `fit()` and every step are pure functions: the same input gives the
   same output, the input is untouched. A summary is a `Compaction` record with the range it covers, kept with
   the steps' cut points in a `ContextState` the caller stores; the originals stay where the caller keeps them
   (design §5.4, §6). A stale state (history changed before a cut) raises `StaleState`, never applies silently.
2. **Explicit, never hidden.** lite-context is called by the host; it is never a hook or middleware that
   rewrites requests on the way to a model. What is sent must be reproducible from what the host stored.
3. **Does not fit → `DoesNotFit`.** Only the steps the caller listed may cut anything; when they are exhausted,
   raise with the report. Never drop content silently, never truncate to "make it fit".
4. **The rules of design §5.2 each have a test**: a batch of tool calls and its results stay together; pinned
   messages, the last human turn and thinking blocks are never touched; every change is in the report and
   everything else is byte-identical (checked at run time: `UnreportedChange`); placeholder texts are fixed
   constants.
5. **No model calls in the library.** A step that needs a model (summary, merge) takes a function from the caller,
   so model calls stay in the host's own role layer. The core stays usable inside a Temporal workflow.
6. **The core has zero run-time dependencies** (standard library only) and depends on no message format and no
   lite module. Adapters (`anthropic`, `openai` as plain dicts in the core; `lite_model` as an extra), counters,
   stores and the REST shell sit at the edges. A test checks that importing the core loads no third-party module.
7. **Adapters round-trip losslessly**: `from_messages(to_messages(x)) == x`; unknown blocks become `Opaque` and come
   back byte for byte. Each adapter has a property test.
8. **Python ≥ 3.11.** No 3.12-only syntax (`type X = ...`, PEP 695 generics); CI runs 3.11 and 3.12.
9. **The CLI and the REST shell hold no logic**: they parse input, call the library and serialise.
10. **Tests never call a real model, a paid API or llm-gateway.** `tests/conftest.py` refuses every socket except
    the PostgreSQL test database (`LITECONTEXT_TEST_DATABASE_URL`, schema `lite_context_test`).
11. **Assert content, not success**: the exact messages out, every action and number in the report.
12. **Measure, don't guess**: a default step order or a claim that masking beats summarising comes with a
    measurement (host before / after, or a lite-eval run), recorded in the KB.
13. **Secrets only from environment variables.** Never commit `.env*` except `.env.example`.
14. **Pinned dependencies**: engine libraries (psycopg, FastAPI, tiktoken) at exact versions in the lock;
    lite-model (adapter extra) as a release wheel in step with the workspace pin.
15. **Hosts are not named in this repository.** It is public; the design calls the first hosts A and B.

## Tech stack

Python ≥ 3.11 (CI 3.11 + 3.12), uv (uv_build), pytest (+ hypothesis; pytest-asyncio from C1), ruff, mypy (strict).
Extras: `lite-model`, `tiktoken`, `pg` (psycopg 3), `server` (FastAPI), `eval`.

## Commands

```
uv run pytest                    # C0-C4 (about 7 s: tests/test_growth.py runs 150 random histories); from C5 with --env-file .env
uv run --python 3.11 --isolated --with-editable . --with pytest==9.0.3 --with hypothesis==6.151.9 python -m pytest   # the 3.11 floor
uv run --env-file .env pytest    # everything; PostgreSQL tests need LITECONTEXT_TEST_DATABASE_URL (from C5)
uv run ruff check && uv run ruff format --check && uv run mypy
uv build --out-dir dist && uv venv --python 3.11 /tmp/c && uv pip install --python /tmp/c dist/*.whl && (cd /tmp && /tmp/c/bin/python -I $OLDPWD/scripts/smoke.py)   # the wheel alone
uv run litecontext serve          # the REST shell, loopback :8750 (extra `server`, from C6)
uv run litecontext migrate        # the Transcript tables (extra `pg`, from C5)
```

## Workflow

1. Work stage by stage (design §14). Each stage ends with every check green; every new test seen red
   once (write it first, or break the line under test and watch it fail).
2. Definition of done: ruff, ruff format, mypy, tests pass on 3.11 and 3.12; acceptance items covered by tests;
   design updated if behaviour changed.
3. Conventional Commits (`feat(fit): …`). All code-facing text in English. No commit without the
   owner's go-ahead for that commit; `git add <paths>` only.
4. **Stage close-out — every stage, after its code passes the checks, before its commit and before the
   next stage starts:**
   1. **Simplify**: run `/simplify` on the stage's full diff and apply its fixes.
   2. **Codex review**, read-only, on the same diff, from the repo root:
      ```
      node "$(ls -d ~/.claude/plugins/cache/openai-codex/codex/*/ | sort -V | tail -1)/scripts/codex-companion.mjs" \
        review --wait --scope working-tree        # or --base <commit before the stage>
      ```
      For focus text use `adversarial-review --wait --scope working-tree "<focus>"`. Fallback without
      the plugin: `codex exec -s read-only -o <out.md> "<prompt>" < /dev/null` — the `< /dev/null`
      matters: with a piped stdin `codex exec` waits for stdin to close and hangs in a background shell.
      If Codex is unavailable, say so; never skip silently.
   3. **Fix what holds up**: verify each finding before acting; a bug gets a failing regression test
      first. A rejected finding is written down with a one-line reason (design §14 "阶段记录"), never
      silently dropped.
   4. **Re-run every check** (step 2) so they cover the final code. If step 3 changed code
      substantially, run steps 1 and 2 once more.
   5. Report the checks' output, Codex's findings and what was fixed or rejected; propose the commit.
5. Host adoption (C4) happens in each host's repository under that repository's rules and close-out; a
   change lite-context needs from it comes back here first (library → release → the host's pin).
