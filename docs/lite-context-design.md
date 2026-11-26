# lite-context design document

**Status: v0.3.2 released (2026-11-26): PDF page counts in compressed streams, `CharEstimate.calibrated` (§4, §15);
C0–C7 done (2026-11-26): lite-agent v0.2.5's compactor uses lite-context (C7); v0.3.1 released: public
`render_transcript`; C6 done, v0.3.0 released (2026-11-26, GitHub Release wheel): REST shell `litecontext serve`
(extra `server`); C5 done, v0.2.0 released (2026-11-26, GitHub Release wheel): `Transcript`, `InMemoryStore`,
`PgStore`, `litecontext migrate`; C4: host A adopted, host B trialled then reverted and deferred (2026-11-26);
v0.1.2 (a call id reused by a later batch no longer gets the new result masked); v0.1.1 (the wheel ships
`py.typed`); v0.1.0 released (C3, 2026-11-26, GitHub Release wheel). C2 done: long-document tools (`split` /
`pack` / `halving` / `reduce_tree` / `select`); C1 done: C1b summaries (`plan_compaction` / `accept` /
`acompact`, `SummaryPolicy`, `fit()` applying summaries); C1a `fit()`, the four steps, `ContextState` and cut
points, runtime checks; C0 done: neutral types, Anthropic / OpenAI adapters, counting and budget. Repository
github.com/ArthurGao/lite-context. The first hosts are two Python projects that bring their own LLM layer
(host A and host B below, §12), adopted together in C4.**
When this document conflicts with the code, update it in the same change and say why in the commit.

## 0. Decision log

The reasons for a decision are written only here; other sections cite this one (§0). "owner" means the owner
decided; "proposed" means it was the drafting judgment, followed until the owner overrules it, and then changed
here.

| Date | Decision | Who | Reason |
| --- | --- | --- | --- |
| 2026-11-26 | Build a separate library, lite-context, that only handles "what should the model see on this one call"; projects that call the LLM directly and lite-agent both use it | owner | Most projects do not use lite-agent and call the LLM directly; each hand-wrote its own chat history, tool result and long-document handling (or has none), and context rot and outright window-overflow errors are a shared gap |
| 2026-11-26 | Form: a library first; the REST API is a thin shell on top and holds no logic | owner | Consistent with lite-rag / lite-mem / lite-agent; REST is for non-Python hosts |
| 2026-11-26 | Explicit calls; no lite-model hook and no "every request passes through" middleware | proposed | Rewriting the request quietly at the call layer makes "what the model saw" differ from what the host recorded, so replay, audit and fork all go wrong (KB `llm-app-architecture` §9). lite-cache can be a hook because it does not change the request content |
| 2026-11-26 | The core depends on no message format and not on lite-model: one set of neutral types inside, adapters at the edges | proposed | Both first hosts have their own LLM layer (one uses its own types, one uses Anthropic-format dicts) and neither uses lite-model; a hard dependency on lite-model would force them to change their LLM layer first |
| 2026-11-26 | The original history is never rewritten; compaction is a `Compaction` that records its coverage, kept in `ContextState` together with the truncation cut points and stored by the caller (§5.4) | proposed | KB §9.1 "compaction masks and keeps provenance". The caller already stores the history (database, workflow state), so it can store one more summary record; there is no need to bring in a lite-context store for this |
| 2026-11-26 | If it does not fit, raise (`DoesNotFit`); never drop content silently | proposed | Both first hosts explicitly chose "over the limit means failure" (one `case_too_large`, one `LLMInputTooLarge`) and treat truncation as an error. lite-context only does the trimming the caller declared |
| 2026-11-26 | Steps that need a model (writing a summary, the merge of a map-reduce) take a function from the caller; lite-context never calls a model itself | proposed | Model calls stay in the host's own role layer (workspace rule: every call goes through a role and carries provenance); the core is therefore pure functions and can be used directly inside a Temporal workflow |
| 2026-11-26 | In REST mode the client writes the summary by default: the service returns the prompt for writing the summary, the client calls its own model and hands the summary back. The service calling a role itself is a later option | proposed | The service holds no model credentials and spends no money; model choice and billing stay in each project |
| 2026-11-26 | Python ≥ 3.11, CI runs 3.11 and 3.12 | owner | Host B runs 3.11; supporting 3.11 costs a pure-Python library very little |
| 2026-11-26 | The core has zero runtime dependencies; the lite-model adapter, the tiktoken counter, PostgreSQL and FastAPI are each an extra | proposed | Host B uses pip requirements, and fewer dependencies are easier to adopt; the core is pure data processing |
| 2026-11-26 | Hosts A and B are adopted together in C4 | owner | Validates two adoption shapes at once: masking in a tool loop, and budget + summary for chat history |
| 2026-11-26 | Host B's chat does stateless trimming: `fit()` again on every request, no stored `ContextState`, no summaries, no migration; document split / halving / reduce are not switched yet | owner | Get "it fits" with the smallest change first; cache stability and summaries wait for measurements on long conversations. Host B's document tools count exact numbers with the vendor's async `count_tokens`, and the core's cost is synchronous (§15) |
| 2026-11-26 | Host B is left unchanged for now: the trial was finished and reviewed, then reverted as a whole, and host B stays as it was | owner | No reason given; the trial's patches and what was learned stay in the record in §14 below, for use when it is adopted again |
| 2026-11-26 | The close-out is the same as the other lite modules: `/simplify` → read-only Codex review → fix or write down the rejection reason → re-run every check | owner | Workspace convention |

## 1. Overview

lite-context answers one question: **what should the model see on this one call**. The input is the caller's
full history (or a set of documents or retrieval results) and a token budget; the output is the content to send
this time, plus a report saying what was trimmed, what was compacted, and how many tokens before and after.
The caller still does the sending.

Three kinds of scenario:

- **Multi-turn history**: chat, and hand-written tool loops. The history keeps growing, and old tool results take
  up most of it (§5, §6).
- **Long documents and retrieval results**: a document that does not fit in one call has to be split, grouped
  and map-reduced; retrieval results have to be selected within a budget (§7).
- **Hosts with no storage of their own**: use lite-context's `Transcript` to store the history (§8).

### Sources

- KB `global/llm-app-architecture.md` §9 (event log + pure projection, compaction only masks) and §11 (truncation
  in a hand-written loop).
- lite-agent's S3b compaction (its design §6): summary events, covered ranges, "a later compaction covers from
  the start", cut points that never split a tool call from its result, and a truncated or empty summary counts as
  a failure. lite-context takes these rules out of the event log and turns them into a form that does not depend
  on a log.
- The existing code of the two first hosts (§12): grouping pages by token budget, halving and retrying when over
  the limit, tree reduce, truncating a tool result when it is produced, and "over the limit means failure".
- Public research (reported, not measured in this project): Chroma "Context Rot" (2025) — the longer the input,
  the worse the quality, starting well before the window limit; JetBrains "The Complexity Trap" (2025) — on
  SWE-agent, replacing old observations with a placeholder works as well as an LLM summary at much lower cost.

### Goals

1. One `fit()` covering the usual ways of handling "the history is too long", trimming level by level in the
   order the caller gives, with every step written into the report (§5).
2. Compaction is reversible and auditable: the original text is not changed, and a summary is a record with a
   covered range (§6).
3. Long-document split, grouping, halving retry and tree reduce are shared tools, with the model call passed in by
   the caller (§7).
4. Depend on no message format: Anthropic dicts, OpenAI dicts, lite-model types and a host's own types can all be
   connected (§3).
5. Testable with no network and no spend: the core is pure functions, and where a model is needed the caller
   passes in a function, with a fake in tests.

### Non-goals (v1)

- No model calls (exception: an optional server-side summary later, §9).
- No token-level compression (LLMLingua and the like); if needed it is added later as an optional step of §7
  (§15).
- No table of model windows: the caller computes the budget and passes it in (hosts have their own table or
  configuration).
- No retrieval, embedding or rerank: that is lite-rag. §7's `select` only picks within already-ranked results by
  budget.
- No long-term memory: that is lite-mem.
- No streaming.

## 2. Layers

```
host (Python, own LLM layer) ──► adapter ─────────────┐
host (Python, lite-model) ─────► adapter(lite-model) ─┴──► lite-context core (pure functions) ──► adapter ──► host's LLM layer ──► model
lite-agent (C7) ──► maps events to neutral Messages ──► the same core
non-Python host ──HTTP──► REST shell (§9) ──► the same core
host with no storage ──► Transcript (§8) ──► the same core
```

Dependency direction: the core uses only the standard library. Core = `types`, `count` (`CharEstimate`), `fit`,
`steps`, `compaction`, `report`, `errors`, `docs/`, `adapters/anthropic`, `adapters/openai`, `transcript/`
(without `pg/`), `testing`.
Extras: `adapters/lite_model` (`lite-model`), `TiktokenCounter` (`tiktoken`), `transcript/pg/` (`pg`), `server/`
(`server`); the core does not import them. C0's `sys.modules` check follows this list. lite-context depends on no
lite module.

## 3. Neutral types and adapters

### 3.1 Types

All are frozen dataclasses (written in a way that works on 3.11: `TypeAlias`, `TypeVar`, no PEP 695).

```python
Text(text: str, extra: Mapping[str, object] = {})
ToolCall(id: str, name: str, raw_args: str, extra: Mapping[str, object] = {})
                                 # raw_args verbatim (OpenAI sends a JSON string, possibly malformed);
                                 # .args() parses on demand and may raise
ToolResult(call_id: str, content: tuple[Text | Opaque, ...],
           is_error: bool | None = None,             # None: the format had no error flag
           content_form: ContentForm | None = None, extra: Mapping[str, object] = {})
Opaque(kind: str, raw: object, tokens: int | None = None,
       text: str | None = None, pages: int | None = None)
                                 # kind: "image" | "document" | "thinking" | "unknown"
ContentForm = Literal["string", "list", "null", "absent"]   # the form a vendor's content had
Part: TypeAlias = Text | ToolCall | ToolResult | Opaque

Message(
    role: Literal["user", "assistant"],
    parts: tuple[Part, ...],
    key: str | None = None,      # the caller's stable id (a DB row id, a seq); needed for Compaction (§6)
    pinned: bool = False,        # never masked, dropped or summarised away (§5.2)
    content_form: ContentForm | None = None,
    extra: Mapping[str, object] = {},   # fields the core ignores (cache_control, name ...)
)
```

- `extra` holds fields the core does not care about but that must be carried back unchanged when converting to
  the vendor format: `cache_control`, OpenAI's `name`, `tool_calls: []`, extra fields inside `function`. The core
  does not read it, only carries it; a masked part keeps no original text other than `extra`.
- `content_form` records the original form of content (a string, a list, `null`, the key absent). It is a typed
  field rather than a conventional key in `extra`, and C1's steps keep it with `dataclasses.replace` when they
  replace content. When it is `None` (built by hand), a single `Text` with no `extra` is output as a string, and
  anything else is output as a list.
- **Shared, not copied**: adapters do not deep-copy. `Opaque.raw` and the nested objects in `extra`
  (`cache_control` and so on) are the same object across input, neutral messages and output; this is deliberate
  (large images and PDFs are not copied). Convention: neither the caller nor lite-context mutates them, and to
  change one, copy it first. lite-context itself never mutates its input (every adapter test checks that the
  input is unchanged).
- The system prompt and tool definitions are not `Message`s: they are passed to `fit()` as `fixed` text (or
  directly as a token count), counted in the budget and never trimmed.
- `Opaque` keeps content the adapter does not recognise or should not touch (images, PDFs, thinking blocks,
  vendor-specific blocks) as it is; `raw` is the original object and is output unchanged when converting back.
  `tokens` is its cost; when the caller knows it, it fills it in, otherwise the counter estimates it by kind and
  by the neutral measures the adapter pulled out during conversion (§4): `text` (a thinking block, or the text of
  a text document), `pages` (a PDF's page count, counted once during conversion). `tokens is None` means
  "estimated"; **never run the text counter on `raw`** (base64 would be counted dozens of times too high). The
  report lists every `Opaque` that was estimated.
- **A thinking block cannot be changed**: Anthropic requires the signed thinking block of a turn that has a
  tool_use to be sent back unchanged. No step masks or truncates `kind="thinking"`; they are removed only as a
  whole, before the most recent `Compaction`, under lite-agent §6's rule.
- **Pairing**: a tool call and its result are paired by `id` / `call_id`. Several calls in one assistant message
  are one **batch**, and their results may be spread over several consecutive messages (OpenAI uses one
  `role: tool` message per result). Every step of §5 works on a whole batch and never splits it.
- **Turn**: a turn starts at a user message that has `Text` and no `ToolResult` (something a person said) and
  runs up to the next such message. A user message that holds only tool results does not start a new turn.
  `DropOldest` and `plan_compaction` both use this definition.
- **Incomplete pairing**: it is normal for the last batch's results not to have arrived yet (`fit()` is called
  in the middle of a loop), and it is output unchanged. Apart from that, a call with no result, a result with no
  call, and a second result for the same call are all input errors: `fit()` raises `BrokenPairing`, and does not
  guess or fill in. While a batch still has an unanswered call, the next message must be one that carries tool
  results (both APIs require this), so a batch is never separated in the middle by something a person said, and
  cutting by turn therefore never splits a batch.

### 3.2 Adapters

An adapter is a pair of functions: `to_messages(native) -> list[Message]` and
`from_messages(list[Message]) -> native`.

| Adapter | Location | Notes |
| --- | --- | --- |
| `anthropic` | core (handles dicts only, does not import the SDK) | The Messages API `messages` array (system is already a separate parameter in Anthropic): string content, `text` / `tool_use` (`input` must be a plain JSON object that round-trips unchanged: no non-string keys, tuples, NaN / Infinity) / `tool_result` blocks (a `tool_result`'s content can be a string or a list of blocks, `is_error`), `cache_control`; other blocks (`image`, `document`, `thinking`, `redacted_thinking` …) become `Opaque` |
| `openai` | core (handles dicts only) | The full Chat Completions `messages` list: `system` / `developer` messages are split out and returned as `fixed` and put back in place on the way back; the assistant's `tool_calls` (only `type: "function"` is accepted; `arguments` goes into `raw_args` verbatim; `tool_calls` that is `null` or `[]` is kept as it is, and the SDK's `model_dump()` produces `null`), `content: null` or no content; each `role: tool` message becomes one user message holding a single `ToolResult`, and on the way back it is again one message each |
| `lite_model` | extra `lite-model` | lite-model's turn types; the version follows the workspace's lite-model pin |
| A host's own types | written in the host | About 30 lines; lite-context's docs give an example |

Rule: **round trips are lossless**. For the input range the table above states, `from_messages(to_messages(x))
== x`, and `Opaque` blocks come back byte for byte. Every adapter has a property test for this, and each of the
following cases has a named case: string content and a single text block; `cache_control` on a block and on a
message; a `tool_result` whose content is a list of blocks (text plus image), `is_error`; a malformed
`arguments` string; `content: null` and `""`; several `role: tool` messages of one batch; the `name` field; the
position of `system` / `developer`. Input outside the table (an unknown role and so on) raises
`UnsupportedInput`, with no best-effort conversion. `key` / `pinned` are not in the vendor formats and are set
by the caller after conversion (the `with_keys()` helper); pinning a call pins its whole batch, results included.

OpenAI's tool messages have no error flag, so `ToolResult.is_error` does not appear when converting to the
OpenAI format. The OpenAI adapter's `Converted.fixed_texts()` turns the system / developer messages into the
text `fit(fixed=...)` wants. When a neutral user message mixes tool results with other content, the OpenAI
output is split in the original order: one tool message per result, one user message per run of other content.

## 4. Counting and budget

```python
class Counter(Protocol):
    def text(self, s: str) -> int: ...
    def opaque(self, part: Opaque) -> int: ...       # used when part.tokens is None
    per_message: int                                 # framing overhead per message

CharEstimate(chars_per_token=3.5, opaque_default=1500, per_message=4)   # default, no dependency
TiktokenCounter(encoding="o200k_base", ...)                             # extra `tiktoken`

Budget(limit: int, reserve_output: int)   # limit = the model's input window as the caller decides it
```

- The default counter is an estimate (characters / 3.5), the same as what host A does today; host B can switch
  to tiktoken. A vendor SDK's count_tokens is a network call and does not go into the core (§15). When you need
  exact numbers, use **calibration** (v0.3.2): the host makes one exact count of a whole text itself,
  `CharEstimate().calibrated(text, tokens)` gives an estimate at that ratio (other parameters unchanged), and you
  hand it to `split(counter=...)` and to the `cost` of `pack` / `select` / `reduce_tree` — one network call per
  document, not one per piece; there is no separate async interface. `text()` first rounds the quotient to 9
  decimal places and then rounds up: a calibrated ratio (7 / 3) makes 7 / (7 / 3) come out as 3.0000000000000004,
  which must be counted as 3 (found by a property test); with the default 3.5 every quotient is a multiple of
  1/7, so the result does not change.
- When `tokens` is not `None` it is used directly and the counter is not called (checked in `count_message`, so
  each `Counter` does not have to remember it). Otherwise `CharEstimate.opaque()` looks only at `kind` and the
  neutral measures and never reads `raw`: `image` 1600; with `pages`, pages × 2000; with `text`, count the text;
  anything else (`redacted_thinking`, a PDF whose pages cannot be read, an unknown block) uses the default 1500.
  These numbers are all constructor parameters. The adapters take the measures: Anthropic's `thinking` text, a
  `document` with a text source, a base64 PDF (counting `/Type /Page`); OpenAI's `file` parts whose `file_data`
  starts with `data:application/pdf;base64,`. Page counts are taken in this order (v0.3.2): the `/Type /Page`
  visible in the file; if there is none, inflate the FlateDecode streams (a PDF 1.5 `ObjStm` holds the page
  objects inside) and count again, with a total inflated size of at most 64 MiB (`MAX_INFLATED`, the most a
  decompression bomb can cost) and streams that cannot be inflated skipped; if there is still none, take the
  largest `/Count` among the page-tree nodes (`/Type /Pages`) (an outline's `/Count` does not count); only if
  there is none is it `None`, falling back to the default. No guessing from file size (rule 12). A file with
  incremental updates may be over-counted. All of this affects only the estimate, not the content. Every step is
  linear in the file: a stream is found by the `stream` keyword and the next `endstream` (a newline before it or
  not, either works), `stream` must be preceded by a delimiter (whitespace or `>`), so `endstream` and
  `upstream` do not count as a start. For a page-tree node, the dictionary that contains it is found by nesting
  in the 4 KiB before and after `/Type /Pages`, and only the `/Count` at its own level is read (one inside a
  nested dictionary such as `/Resources << /Count 99 >>` does not count); each search continues from where the
  last one stopped, so a node scans the window once; at most 1000 nodes are looked at (a 9 MB file made entirely
  of nodes takes 0.06 seconds, 0.60 seconds without the cap). Limits (affecting only the estimate): when a
  page-tree dictionary is over 4 KiB (a flat root node with thousands of `/Kids`) or a string contains `>>`, it
  cannot be counted and falls back to the default; a file with incremental updates where one page is visible and
  the rest are in compressed streams is counted by the visible one (the same as v0.3.1).
  The first version used two regular expressions, and review found them quadratic even on small input (68
  seconds for 280 KB); the second version's dictionary search rescanned the whole window at every step, taking 6
  seconds on 0.07 MB of constructed input — both were fixed, and the worst constructed inputs now all finish in
  under 0.1 seconds.
- `CharEstimate`'s parameters must be positive (`chars_per_token`) or not negative (the others); a token count in
  `fixed` must be a non-negative int (`bool` does not count), otherwise `ValueError`.
- `TiktokenCounter` is postponed to C4, when host B is adopted: tiktoken downloads its encoding file from the
  network on first use, which conflicts with "tests do not touch the network", and how the encoding file ships
  with the package has to be decided first; host B can also implement the `Counter` protocol directly.
- The budget decision uses estimates, and accounting uses the usage the vendor returns (host A's decision D-10
  does the same). Numbers in the report are marked as estimates.
- Available amount = `limit - reserve_output - fixed`. If `fixed` alone is over the limit, raise `DoesNotFit`
  straight away without any trimming.

## 5. History: `fit()`

```python
fitted = fit(
    messages,                          # the caller's full history, never mutated
    budget=Budget(limit=200_000, reserve_output=8_000),
    fixed=[system_prompt, tools_json], # counted, never cut
    steps=[TruncateToolResults(max_tokens=2_000), MaskToolResults(keep_last=3), DropOldest()],
    state=saved_state,                 # optional, from the caller's storage: compaction + cut points (§5.4)
    counter=CharEstimate(),
)
fitted.messages        # what to send this turn
fitted.state           # store it next to the history; pass it back next call
fitted.report          # tokens before / after, every action taken, compaction_due
# raises DoesNotFit(report) when every step has run and it still does not fit
```

### 5.1 Flow

1. If there is a `state.compaction`, apply it first (§6.2): the covered messages are replaced by one summary
   message.
2. Replay the previous cuts using the cut points recorded in `state` (§5.4).
3. **Standing steps** (`always=True`, such as `TruncateToolResults`) run on every message, whether or not the
   budget is exceeded. Their result for a given message is always the same, so they do not break the prompt
   cache prefix.
4. If it is still over the high-water mark, run the **cutting steps** in order. Each step cuts until the
   low-water mark (`target`, 70% of the budget by default) and is skipped if the total is not over; the new cut
   points are written into the returned `state`.
5. If it is still over budget after everything has run → `DoesNotFit`; the report says what each step did.
6. The high-water mark is 90% of the budget by default. `report.compaction_due` is computed **after** steps 1–4:
   it is true when, with the existing summary and all cuts applied, the total is still over the high-water mark,
   and it tells the caller that a new summary should be written (§6).

### 5.2 Invariant rules (each has a test)

1. The input `messages` and `state` are not modified; the same input gives the same output (pure function).
2. A batch's `ToolCall`s and their `ToolResult`s are either all present or all absent; masking replaces only the
   content of a result and never deletes the pairing. Incomplete pairings are handled as in §3.1.
3. `pinned` messages are never masked, dropped or covered by a summary.
4. The human message of the last turn (the last user message that has `Text` and no `ToolResult`) is never
   touched. The newest tool result is **not** covered by this protection: a standing step may truncate it and
   write that into the report; otherwise one oversized result would make every call `DoesNotFit`.
5. The report lists every message that was touched (`key`, or the position when there is no key), the action and
   the tokens saved; a message that is not listed is byte-for-byte the same as in the input. This is checked at
   runtime: after each step, `fit()` compares the step's input and output, and if a message was changed or
   removed but is not in that step's report it raises `UnreportedChange`. Steps written by a host are subject to
   the same check.
6. Every placeholder is a fixed constant written in the step, and tests compare it byte for byte.
7. An `Opaque` with `kind="thinking"` is not changed by any step (§3.1).

### 5.3 Steps

| Step | Type | What it does |
| --- | --- | --- |
| `TruncateToolResults(max_tokens, keep="head" \| "head_tail")` | standing | Cuts an over-long tool result to the limit and adds `[cut: N more characters not shown]`; it does not cut when adding the marker would make it longer. The limit is applied to the whole result: when a result has several pieces of text and it is over, they are joined with newlines into one piece (position and `extra` taken from the first piece) and then cut; the tail of `head_tail` is taken only from what the head leaves, so the two parts never overlap |
| `MaskToolResults(keep_last=N)` | cutting | Starting from the oldest **batch**, replaces the content of every tool result in the batch with `[result cleared to save context; call {name} again if you need it]`, leaving the most recent N batches alone; images in a result are replaced too |
| `MaskOpaque(keep_last=N)` | cutting | Old `image` / `document` blocks (in messages, and in tool results) are replaced with `[{kind} removed to save context]` (what host B does for attachments today); `thinking` is not touched. The unit is "one message that carries an image or document", and everything in one message is cut together; `keep_last` counts the most recent such messages |
| `DropOldest(keep_last_turns=N)` | cutting | Deletes the oldest whole turn (the definition in §3.1, so a batch is never split) and leaves `[N earlier messages not shown]` at the position of the first deleted message (N is the number of deleted messages). A turn that contains a protected message is not deleted. It is not in the steps by default; it is added only when the caller asks for it explicitly |

No step touches a protected message (a pinned one, or the human message of the last turn).

**A host's own steps** (the protocol C1a implements): a step works on `Entry(ref, message)`, where `ref` is the
caller's key, or the position `#i` in the input when there is no key, and a newly inserted message uses its own
ref (such as `+drop_oldest`).

- Standing step (`AlwaysStep`): `name`, `always = True`, `apply(entries, ctx) -> (entries, actions)`.
- Cutting step (`CuttingStep`): `name`, `always = False`, `units(entries, ctx) -> [Unit]` (the units that can be
  cut, oldest to newest; `Unit.refs` are the messages it covers; `Unit.at` is the step's own locator),
  `apply(entries, units, ctx)` (cuts only the units it is given). `fit()` decides how many to cut: it first
  replays the units at and before the cut point in `state`, and when still over the high-water mark it adds more
  going forward until the low-water mark. When adding it stops only at message boundaries (the several units of
  one message are cut together, because a cut point records a whole message); it does not assume "the more cut,
  the more saved" (a placeholder can be longer than the original): it cuts all the units at once, works out from
  the messages that were touched how much each unit saves, estimates the total for each count with a running
  sum, takes the smallest count that reaches the low-water mark (or, if none does, the count that saves the
  most); then it really cuts once more and counts exactly, and if the estimate was too low it tries going
  further. If none of this beats the replay, the replay is kept.
- `StepContext` gives the counter and the protected refs.
- After each step `fit()` checks: protected messages and thinking blocks are unchanged (`ProtectedChange`);
  messages not reported are all present, unchanged and in the same order, and every added message was reported
  (`UnreportedChange`); pairings are still complete (`BrokenPairing`).
- Every candidate output is checked, including the ones `fit()` tried during its binary search and did not end
  up using, so a step that breaks a rule is always found.
- `lite_context.testing.check_step(step)` puts a host's step into `fit()` and runs it over several sample
  conversations (parallel batches, an unfinished batch, pinned, images, thinking) and several budgets, checking
  the rules above, that two calls give the same result, and that the input is unchanged; then it grows each
  conversation one turn at a time (three budgets), passing `state` back each time, and checks that on the calls
  where the cut point did not move, the previous request is a prefix of this one (the cache is not broken; the
  message that just left protection is the exception, §5.4); then it runs once more on a history without keys.
  It returns the number of times the step actually changed something. The samples include a lone image, two
  images in one message, and a parallel batch where each result carries one image.

### 5.4 State: why the cuts are recorded

`fit()` is a pure function and looks only at its input. If it does not record how far the previous call cut,
then once the raw history goes over the high-water mark, every later turn would cut down to the low-water mark
again, and since the history grows every turn, every turn would mask one more old message, and the prompt cache
would be invalidated from that point every turn. So the result of each cutting step is a **cut point**: which
message in the input it cut through (`through`), plus a digest of the refs of the input from the start up to
that message. On replay, the step cuts all its units at and before `through`. A cut point is recorded by
position, not as "the first few units": earlier steps may have deleted some units, or a message may have left
protection (the human message of the last turn was replaced by another), so the list of units changes but the
position does not, and so `StaleState` is not raised by mistake (found in C1a review). The cost: the old image
that left protection is cut at the next replay, so an old message changes once because of that. `fit()` orders
the units by their position in the input and does not depend on the order in which a step lists them. The cut
points are kept in `ContextState`:

```python
@dataclass(frozen=True)
class ContextState:
    cuts: Mapping[str, Cut]       # step name -> Cut(through: ref, digest of the input refs up to it)
    compaction: Compaction | None # C1b
```

- Pass `state` back on the next call: existing cut points are replayed unchanged; a cut point moves forward
  only when the total is still over the high-water mark after the replay. A cut point only moves forward, never
  back.
- If a cut point moved forward would make the total larger instead (`DropOldest`'s placeholder is longer than
  the few short messages it deleted), it is not adopted and the replay result is kept.
- The caller stores `state` next to the history (`to_json()` / `from_json()`, one JSON column is enough; the
  format carries a version number, and an unknown version is an error). Not passing `state` is the same as
  starting from zero: the result is still correct, only the cache is not stable.
- Like `Compaction`, a cut point carries a digest of the keys of the messages it covers; when the history was
  changed before the cut point, `StaleState` is raised (§6.2).
- **A cut point measures the history after the summary is applied.** The `state` returned by a new compaction
  (§6.1 `accept`) carries no cut points: the units covered by the summary are gone and the old cut points'
  digests would not match; also, compaction itself already invalidates the prompt cache from the start, so
  starting to cut from zero loses nothing (decided at the close of C1a, see the stage records).
- A key cannot start with `#` or `+` (those are lite-context's internal refs), otherwise `ValueError`.
- Cut points depend on keys: when messages have no key `fit()` can still cut, but the returned `state` records
  no cut points; a `state` carrying cut points paired with messages that have no key raises `StaleState`. A cut
  point for a step that is not among this call's steps is not carried into the new `state`.

## 6. Summary: `Compaction`

```python
@dataclass(frozen=True)
class Compaction:
    covers_through: str      # key of the last message the summary covers
    covered_digest: str      # sha256 over the covered messages' keys, in order (after the pinned prefix)
    text: str                # exactly what the model is shown, preamble included
    provenance: Mapping[str, object]   # role / vendor / model / usage, as the caller's LLM layer reports them
    created_at: str          # ISO 8601
```

A `Compaction` lives in `ContextState.compaction` and is serialised together with the cut points (`to_json()` /
`from_json()`).

### 6.1 Writing the summary

```python
if fitted.report.compaction_due:
    plan = plan_compaction(messages, state=saved_state, policy=SummaryPolicy(keep_last_turns=4))
    if plan is not None:                     # None: nothing new can be covered
        reply = my_llm.call(role="compact", system=plan.system, messages=plan.prompt)  # the caller's role
        new_state = plan.accept(reply.text, truncated=reply.stopped_at_limit, provenance=reply.provenance)
        store(new_state)                     # raises EmptySummary / TruncatedSummary instead
```

- `plan.covered` is the keys of the messages the summary will stand for (counted from after the pinned prefix),
  and `plan.covers_through` is the last one; the end of the covered range is always the start of some turn
  (`keep_last_turns` is at least 1, and the current turn and an unfinished batch never go into the summary; if
  there are no more turns than the number to keep, or a history with no human messages, None is returned).
- `accept` stores `provenance` as it looks after a JSON round trip (tuples become lists); a value that is not
  JSON raises `ValueError` at once, rather than failing only when it is stored.
- The convenience function `await acompact(messages, state=..., summarize=fn, policy=...)` combines the three
  steps; `fn(system, prompt)` is an async function from the caller and returns `Summary(text, truncated,
  provenance)`; it returns None when there is nothing new to cover.
- The `state` returned by `accept` has only the new summary and no cut points (§5.4).
- **Covered range**: it starts after the pinned prefix at the start of the history (for example the initial task
  statement) and ends at the cut point. There can be no pinned message in the range: the cut point goes at most
  as far as before the next pinned message, so pinned messages always stay where they were, in the same order.
- The cut point falls only on a turn boundary (§3.1), so it never splits a batch. A later compaction covers from
  the start and takes the previous summary in too; a truncated or empty summary counts as a failure and cannot
  stand in for the history. These rules follow lite-agent S3b.
- **`SummaryPolicy` is replaceable**: `keep_last_turns`, `render(previous, messages) -> (system, transcript)`,
  `wrap(mark, reply) -> text`, `unwrap(text)`. `mark` is fixed once per plan and recorded on the plan, and `wrap`
  removes only the header line that carries this mark (a mark can be any string); the previous summary is given
  to the model with its preamble removed, so the preamble is not copied into the new summary and wrapped
  another layer deep. `render_transcript(mark, messages)` (v0.3.1) gives the part without the closing `END`, for
  hosts that assemble their own prompt (lite-agent's compactor, C7). The default policy copies what lite-agent
  measured (its §15 S3b): every segment of the transcript starts with a header that carries a random token (text
  inside a tool result cannot impersonate the user); the instructions are said again after the transcript (a
  small model otherwise answers the last question in the transcript, 6 times out of 6); header lines copied into
  the reply are removed; thinking blocks do not go into the transcript and images are written as `[image]`; the
  previous summary goes in as the first segment, `earlier summary`. `mark` can be passed in (for tests) and is
  random each time by default. lite-agent provides its own policy in C7, so that it keeps its current behaviour.
- `key` is required: a history without keys cannot be compacted (an error). A key only has to be unique within
  one history and **does not have to be comparable** (a UUID is fine).

### 6.2 Applying the summary

`fit(..., state=s)`, `s.compaction = c`. Applying the summary is an internal step in `fit()`: like other steps it
reports its actions (the covered messages `summarised`, the summary message `inserted`) and goes through the
same runtime check, with no separate set of rules:

1. Find the position of `c.covers_through`; not found → `StaleState`.
2. For the messages from after the pinned prefix up to this position, the digest of their keys must equal
   `c.covered_digest` (if the count changes the digest changes too, so the count is not stored separately;
   `covered_count` was removed at the close of C1b); if not equal (a message before the cut point was inserted,
   deleted or had its key changed) → `StaleState`. A stale summary is never applied quietly.
3. This position must be a turn boundary and there can be no pinned message in the range; otherwise →
   `StaleState`.
4. Output: the pinned prefix unchanged, then one user message (its content is `c.text`, ref `+compaction`),
   then the messages after the cut point. Only the most recent `Compaction` is used. Later cutting steps work on
   this result, so the cut points measure the history after the summary (§5.4).
5. If the messages have no key but `state` has a summary → `StaleState`.

The summary message is a user message, usually followed directly by the next human message (also user), and when
there is a pinned prefix before it there are more: consecutive user messages. Anthropic merges them and OpenAI
accepts them; backends that require strict alternation (the chat templates of some local models) need the host
to merge them itself.

A change to the **content** of a message before the cut point (with the key unchanged) cannot be detected; this
is a convention on the caller: messages covered by a summary are not modified again. To change one, drop the
`state` and compact again.

## 7. Long documents and retrieval results: `lite_context.docs`

Things both first hosts wrote for themselves. The model call is always passed in by the caller. The rules come
from KB `global/map-reduce-llm-guide.md` §3 and the hosts' measurements: count first, and if it fits make one
call with the text as it is (on the split path the prompt and the call are exactly the same as without a split);
reduce groups adjacent notes by budget and merges them level by level; a single note that cannot be merged, or a
merge result over budget, makes the whole reduce fail, and it never falls back to plain concatenation; only a
failure the caller judges to be "too large, rejected" is halved, and any other failure is raised as it is.

| Function | What it does |
| --- | --- |
| `split(text, max_tokens=, counter=, overlap=0.06, prefer="\n", heading=None) -> [Piece]` | If it fits, one piece as it is. Otherwise break after each `prefer` into small segments (an over-limit segment is first cut down to the limit), counting each segment once; one piece takes as many whole segments as possible and is then checked with one real count (a counter that is not additive, if over, backs off from the tail); the next piece overlaps backwards by about `overlap`, also starting at a segment boundary. Adding up the segment counts overestimates for a subword tokenizer, which is the safe direction (KB guide §3); the total counting work is about 3 times the full text. `heading` is a regular expression: each piece carries the heading line of the section it starts in (if it starts on a heading line, that heading; host B measured: without it, 60 of 60 pieces lost their section after the split) |
| `pack(units, budget=, cost=, overhead=0) -> [[unit]]` | Groups consecutive units by budget without reordering them; `DoesNotFit` when a single unit is over budget by itself |
| `await halving(items, call, too_large=, min_size=1) -> Halved(calls)` | Calls `call(items)`; when `too_large(exc)` is true, halves and calls the two halves one after the other; any other failure, a rejection that is still there at `min_size`, and cancellation (`BaseException`, not given to `too_large`) are raised as they are. `calls` records in order the range each successful call read and its result, and `results` is the joined result. `min_size` is at least 1. Text is split by line: pass `text.splitlines(keepends=True)` |
| `await reduce_tree(notes, budget=, cost=, merge=, overhead=0, max_depth=8) -> Reduced(notes, levels)` | Merges adjacent notes level by level until they fit in the budget together; the returned notes are read by the caller with its own final prompt (the prompts of the intermediate merges and of the final reduce differ). The groups of each level are merged one at a time, in order (concurrency on a rate-limited account only gets rejected). No progress, a merge result over budget, or reaching `max_depth` all raise `DoesNotFit` |
| `select(chunks, budget=, cost=, key=) -> Selected(chosen, duplicates, over_budget)` | Takes from retrieval results the caller has already ranked, in order, and de-duplicates by key; it stops at the first one that does not fit, and every non-duplicate after that goes into `over_budget` and duplicates go into `duplicates` (a short low-ranked chunk never displaces a high-ranked one; the three lists exactly partition the input) |

Units, notes and retrieval chunks are all generic: the caller uses its own types (a page, `(range, text)`, a
retrieval result) and supplies `cost`.

## 8. `Transcript`: for hosts with no storage

```python
store = PgStore(url)                                       # or InMemoryStore() (extra `pg` for PgStore)
t = Transcript(store, "thread-1")
await t.create()                                           # creating again does nothing
seq = await t.append([user_message], expected_seq=await t.last_seq())
fitted = await t.project(budget=..., steps=[...])          # fit() over the messages + the latest state
reply = await my_llm.call(fitted.messages)
seq = await t.append([assistant_message, fitted.state], expected_seq=seq)   # the state is an entry too
await t.compact(summarize=fn)                              # acompact() + appends the new state; None when nothing is due
```

- **Async** (decided in C5): the methods of the store and of `Transcript` are all `async`. Every likely caller is
  async (C6's FastAPI shell, async hosts, lite-agent), and this is consistent with lite-agent's `SessionStore`.
  `project()` itself is still a pure function; it just has to read the store first.
- Append-only: an entry is a `Message` or a `ContextState`, carrying a `seq` that increases monotonically from 1
  within each transcript. **A message's key is its seq** (`"1"`, `"2"` …), assigned by `Transcript`; a message
  that arrives with a key is rejected (`ValueError`), not quietly overwritten. This way the cut points in a
  stored `ContextState` always point at the same message.
- `append(items, expected_seq)`: when `expected_seq` is not the current tail it raises `SeqConflict` and writes
  nothing; a batch of entries is all or nothing; the seqs in the batch must follow on from the tail. Under
  concurrency one succeeds and the rest are rejected (in PG this relies on `FOR UPDATE` on the transcript row).
- `project()` is a pure projection: all messages sorted by seq, plus the latest `ContextState` (`None` when there
  is none), handed to `fit()`. The same entries give the same output; when `fitted.state` has changed, the
  caller appends it back.
- `compact()`: runs `acompact()` over the same messages and latest state, appends the new state and returns its
  seq; if someone appended while the summary was being written it is `SeqConflict` (just run it again).
- The store keeps JSON records: messages use `lite_context.codec` (`message_to_dict` / `message_from_dict`,
  versioned, keeping every field of every part) and state uses `ContextState.to_json`. Writing is checked by the
  standard of reading back: content that JSON cannot restore to an equal value (tuples, non-string keys, NaN,
  objects), fields of the wrong type (`pinned=1`, `tokens=True`, a tool result nested in a tool result), and a
  NUL character or a lone surrogate anywhere (PostgreSQL cannot store them) are rejected at write time — a
  transcript can only be appended to, so something that is written and cannot be read back is broken for good.
  The in-memory store and the PG store therefore behave the same.
- PostgreSQL (extra `pg`, psycopg 3.3.8, the same version as lite-agent): database and role `lite_context`,
  schema `lite_context`; only a schema named `lite_context` or `lite_context_<name>` is accepted. Tests use only
  the schema `lite_context_test` in the same database, with the URL from `LITECONTEXT_TEST_DATABASE_URL`, and
  when it is not set the tests fail rather than skip. Tables `transcripts` and `entries`; `entries.data` is
  `json` and not `jsonb` (jsonb reorders the keys of an object, so a vendor block the adapter gives back as it
  was would change its bytes and the prompt cache prefix would break); `entries` has triggers that reject UPDATE
  / DELETE / TRUNCATE (the same as lite-agent's events table). Any database failure comes out as `StoreError`
  (carrying only the exception type and the SQLSTATE); after a first connection failure the next call connects
  again. Migrations are the package's `NNNN_*.sql`, run by `litecontext migrate` / `litecontext pending` (URL
  from `LITECONTEXT_DATABASE_URL`). A connection error shows only the exception type and the URL with the
  password removed.

## 9. REST shell

`litecontext serve` (extra `server`, FastAPI 0.162.2 + uvicorn 0.54.0, the same versions as lite-agent) binds
only to the loopback address (default 127.0.0.1:8750; `0.0.0.0`, a LAN address, and even the name `localhost` are
all refused) and does no authentication, like lite-agent's HTTP shell; that is decided when a host needs to
reach it from another machine. A Host that is not a loopback name gets 400, a foreign Origin gets 403, a POST
that is not JSON gets 415 (these three, and `check_host`, are copied from lite-agent); a request body over
`LITECONTEXT_MAX_BODY` (default 32 MiB) gets 413 — a declared length that is over is refused at once, and an
undeclared one is counted as it is read. The shell has no logic: `schemas` parses (an unknown field gives 422),
`wire` converts, `app` calls the library once and serialises, so the same input gets the result the library
gives.

| Endpoint | State | Maps to / returns |
| --- | --- | --- |
| `POST /v1/fit` | stateless | `fit()` → `{messages, state, report}` |
| `POST /v1/compaction/plan` | stateless | `plan_compaction` → `{system, prompt, covered, mark, keep_last_turns}` or `null` |
| `POST /v1/compaction/accept` | stateless | hand the plan back as it was and add `reply` (`truncated`, `provenance`, `now` optional) → `{state}` |
| `POST /v1/docs/split` | stateless | `split` → `{pieces: [{start, end, text, heading}]}`; `heading` is a regular expression (≤200 characters, matched per line) |
| `POST /v1/docs/pack`, `/v1/docs/select` | stateless | units `{text?, tokens?, key?}`, returns indexes: `{groups}` / `{chosen, duplicates, over_budget}` |
| `POST /v1/transcripts` | stateful | `{id}` → `{id, last_seq}` (creating again is harmless) |
| `POST /v1/transcripts/{id}/entries` | stateful | `{format, items: [{message} \| {state}], expected_seq}` → `{last_seq}` |
| `GET /v1/transcripts/{id}/entries?after=&format=` | stateful | `{entries: [{seq, kind, item}]}`; in OpenAI format a message's `item` is a list of chat messages |
| `POST /v1/transcripts/{id}/project` | stateful | `Transcript.project` → the same shape as `/v1/fit` |
| `POST /v1/transcripts/{id}/compaction/plan` / `.../accept` | stateful | `Transcript.plan_compaction` → the plan plus `expected_seq`; accept appends the state after that seq → `{last_seq}` |

- Messages carry `format: anthropic | openai | neutral` and the same format comes back; neutral is the `codec`
  record. A leading system / developer message in OpenAI format counts as fixed and is put back at the very
  front as it was; it cannot be stored in a transcript (422, pass it as fixed text). A summary needs every
  message to have a key: for the Anthropic / OpenAI formats use the optional `keys`.
- Steps: `[{"step": "truncate_tool_results", "max_tokens": N} | {"step": "mask_tool_results", "keep_last": N} |
  {"step": "mask_opaque", "keep_last": N} | {"step": "drop_oldest", "keep_last_turns": N}]`. The counter is fixed
  as `CharEstimate`.
- Errors: `{error, detail}`, where `error` is the exception name in snake case. `DoesNotFit` is 422 and carries
  `report`; `UnsupportedInput`, `BrokenPairing`, an empty or truncated summary, and input the library rejects are
  422; `StaleState` and `SeqConflict` are 409 (the latter carries `expected` and `actual`); `UnknownTranscript`
  is 404; `StoreError` is 503; the transcript endpoints are 503 when there is no database.
- Transcripts use `PgStore` (`LITECONTEXT_DATABASE_URL`, connected on first use after start-up and disconnected
  on shutdown); tests can inject `InMemoryStore`. `halving` / `reduce_tree` need callbacks and are not in REST.
  The server calling a role itself to write the summary (which needs lite-model and credentials) is a later
  option (§15).

## 10. Configuration

A library's parameters are all passed in code. The REST shell's configuration comes from environment variables
with the prefix `LITECONTEXT_` (`DATABASE_URL`, `HOST`, `PORT`, `MAX_BODY`);
`.env.example` is the template, and `.env` does not go into git.

## 11. Tests

- pytest; `tests/conftest.py` refuses every socket (except the PostgreSQL tests, which connect only to
  `LITECONTEXT_TEST_DATABASE_URL`).
- Every rule of §5.2 has at least one named test; adapters have a round-trip property test (hypothesis).
- Assert concrete content: every message in the output, every action and number in the report, not only "no
  error".
- CI: ruff, ruff format, mypy strict, pytest, each run on Python 3.11 and 3.12.
- Measuring the effect (from C4): before and after a host is adopted, record on the same batch of real calls the
  input tokens, the number of window-overflow failures and the prompt cache hits; the quality comparison of
  masking and summaries is run with lite-eval (§15).

## 12. Relation to other projects

| Project | Relation |
| --- | --- |
| Host A (Python 3.12, uv, its own LLM layer and types, a tool loop inside Temporal) | C4: each step of the tool loop calls `fit()` first (`TruncateToolResults` + `MaskToolResults`), replacing today's truncation only when a result is produced; a pure function, it stays in the workflow; the summary goes through a new activity. Page grouping and halving retry for documents are replaced by §7 |
| Host B (Python 3.11, pip, Anthropic-format dicts, chat history in PostgreSQL held by a TypeScript service) | C4 deferred (owner 2026-11-26): a trial called `fit()` first when a chat request entered Python (`MaskOpaque` + `DropOldest`, stateless) and was reverted; the document tools are not switched (§15) |
| lite-agent | C7: depends on lite-context, maps events to neutral `Message`s, and its `SummaryCompactor` and the masking rules in its projection switch to lite-context. Its design needs to change (rule 8: lite-model is the only lite dependency) |
| lite-model | Not a dependency. The `lite_model` adapter is an extra, and the version follows the workspace pin |
| lite-rag, lite-mem | Not a dependency. lite-rag handles retrieval and ranking, and §7's `select` only picks within the results it has ranked |
| lite-eval | After C4, used to compare the quality of masking and summaries (extra `eval`) |

Installation: GitHub Release wheel (the same as lite-model and lite-agent). A uv host writes the wheel URL in
`[tool.uv.sources]`; a pip host writes `lite-context @ <wheel URL>` in its requirements.

## 13. Project structure

```
lite-context/
  src/lite_context/
    types.py, count.py, fit.py, steps.py, compaction.py, report.py, errors.py
    adapters/      _common.py, anthropic.py, openai.py, lite_model.py (extra)
    docs.py        split, pack, halving, reduce_tree, select
    transcript/    store.py, memory.py, pg/ (extra `pg`)
    server/        app.py (extra `server`)
    cli.py         litecontext serve | migrate
    testing.py     invariant fixtures for hosts' own steps
  tests/
  docs/lite-context-design.md
```

## 14. Stages and acceptance

Every rule of §5.2 and §6 must map to at least one named test; when a stage is closed out, the "rule → test"
mapping is listed.

Every stage closes out by step 4 of the AGENTS.md Workflow: `/simplify`, then a read-only Codex review, then
fixes (rejected findings are written in this section's stage records), and finally re-run every check.

| Stage | Content | Acceptance |
| --- | --- | --- |
| S0 | Repo, design document, AGENTS.md / CLAUDE.md / README, project KB, registration in the workspace `AGENTS.md`, registration of port 8750 | This document confirmed by the owner; Codex's review of the design handled |
| C0 | Skeleton: pyproject (uv, uv_build, ruff, mypy strict, pytest, `requires-python >=3.11`), CI (3.11 + 3.12), a conftest that refuses sockets; §3's types and the `anthropic` / `openai` adapters; §4's counter and budget; `Report`, `DoesNotFit` | ruff, mypy and pytest all green on both versions; the adapter round-trip property test plus every named case listed in §3.2; input outside the table raises `UnsupportedInput`; an `Opaque` estimate goes by kind and never counts `raw`; the core has no third-party module at import time (the test checks `sys.modules` against §2's list); `fixed` over budget goes straight to `DoesNotFit` |
| C1 (two steps: C1a = `fit()`, steps, `ContextState` and cut points, runtime checks, `testing.check_step`; C1b = §6's summaries) | `fit()` and §5.3's four steps, high and low water marks, `ContextState` and cut points (§5.4), §6's `plan_compaction` / `accept` / `acompact`, `SummaryPolicy` and applying the summary | Each of the seven rules of §5.2 has a test (including `UnreportedChange` catching a hand-written step that deliberately does not report); two `fit()` calls on the same input give the same result; 20 consecutive calls with `state` while the history grows each turn: old messages change only in the turns that cross the high-water mark, and in the other turns the output has the previous turn's output as a byte-for-byte prefix (cache stability); the three incomplete-pairing cases (an unfinished last batch passes, a missing result and a missing call raise `BrokenPairing`); `keep_last` counts by batch; `DropOldest` does not split a turn that starts with a tool result; an oversized newest tool result is truncated by a standing step and not `DoesNotFit`; no step changes a thinking block; the three `StaleState` cases (cut point not found, count or digest mismatch, cut point not on a turn boundary); after a summary the original messages are unchanged and the output starts with the summary; a later compaction takes in the previous one; a truncated or empty summary fails; a cut point does not split a pairing, does not cross a pinned message, and the pinned prefix stays before the summary; a `covers_through` that cannot be found is an error; the `testing.py` fixture can be applied to a host's own step |
| C2 | §7: `split`, `pack`, `halving`, `reduce_tree`, `select` | Tests written from the behaviour of the two first hosts' existing implementations (overlap ratio, cutting at newlines, an error when a unit is over budget, raising after halving down to `min_size`, number of levels, de-duplication); `reduce_tree` raises, rather than looping forever, when `merge` does not shorten the input, when a single note is over budget, and at `max_depth`; the caller's fake records the input of each call, and the tests assert grouping and order |
| C3 | Release the v0.1.0 release wheel | A fresh 3.11 pip environment and a 3.12 uv project can both install from the wheel and run the README example |
| C4 | Hosts A and B adopted together (done in each one's own repo, each closes out on its own); on the lite-context side, only fix the problems found during adoption | All tests of both hosts pass; the input tokens, window-overflow failures and cache hits of the same batch of calls before and after adoption are written into the KB; host B's `ContextState` persistence has a migration and tests; **pass condition**: replaying a batch of real requests recorded before adoption gives 0 window-overflow failures, input tokens no higher than before adoption, and each host's own tests all green |
| C5 | §8: `Transcript`, `InMemoryStore`, `PgStore` (extra `pg`), `litecontext migrate` | Append-only (a trigger rejects UPDATE / DELETE); with concurrent appends one succeeds and one is rejected; the projections of the in-memory and PG stores are the same; the projection is the same after reading back across processes |
| C6 | §9 REST shell (extra `server`) | Every endpoint's output matches the library function's field by field (the same set of fixtures run twice); binds only 127.0.0.1; an over-limit request body is rejected |
| C7 | lite-agent switches to lite-context (in the lite-agent repo, plan mode first): it provides its own `SummaryPolicy` (how turns are counted, rendering, header handling) | All of lite-agent's compaction tests still pass without changing a line; where that cannot be done, first add the capability in lite-context, or write down in lite-agent's stage records, one by one, why a test was changed; its design §6 and rule 8 are updated together |
| As needed | Token-level compression step, server-side summaries, counters other than tiktoken, an async counter | — |

### Stage records

(Appended at the close of each stage: what was done, the rule → test mapping, Codex's findings and how they were
handled.)

- **S0 (2026-11-26)**: repo, first draft of this document, AGENTS.md / CLAUDE.md / README, project KB, workspace
  registration, port 8750. Codex was unavailable (account quota used up, back only on 2026-12-18), so a Claude
  subagent that did not have the conclusions of this document's author did an adversarial design review; all 12
  findings held and all were accepted:
  1. A pure-function `fit()` with high and low water marks cannot keep the cache stable → add `ContextState` and
     cut points (§5.4).
  2. Cutting "turns" by user message would split a batch that starts with a tool result → redefine the turn
     (§3.1).
  3. "Do not touch the last user message" conflicts with standing truncation → protect only the human message of
     the last turn (§5.2-4).
  4. The neutral types cannot hold every field of the vendor formats → `raw_args`, result content that can hold
     images, `extra`, and state the round-trip range and named cases (§3).
  5. Masking a thinking block would be rejected by Anthropic → no step changes a thinking block (§3.1, §5.2-7).
  6. Estimating `Opaque` → estimate by kind, never count `raw`, mark it in the report (§3.1).
  7. Keys that are not comparable, a summary going stale after the history is changed, pinned inside the covered
     range → `covered_count` + digest + `StaleState`, and a covered range with no pinned message (§6).
  8. Incomplete pairing was not defined → written into §3.1, `BrokenPairing`; `keep_last` counts by batch.
  9. A hand-written step could quietly lose content → runtime `UnreportedChange` check (§5.2-5).
  10. `reduce_tree` might not converge → termination conditions (§7).
  11. C7 inconsistent with lite-agent's summary rules, C4 had no pass condition → a replaceable `SummaryPolicy`,
      and C4 states its pass condition (§6.1, §14).
  12. Small contradictions (`expected_seq`, the REST shell lacking compaction endpoints, the test schema, the
      core module list, when `compaction_due` is computed) → corrected one by one.
- **C0 (2026-11-26)**: `pyproject` (uv_build, consistent with the other lite modules; the hatchling in the
  original text was changed; `requires-python >=3.11,<3.13`, ruff `target-version = py311`, mypy
  `python_version = 3.11`), CI (3.11 + 3.12 matrix), `types` / `count` / `report` / `errors`,
  `adapters/anthropic`, `adapters/openai`. Deviations from this document: the `Opaque` estimate lives in the
  counter rather than the adapter (§3.1 and §4 changed; at the close-out it was changed back to the adapter
  taking the measures, see /simplify below); `TiktokenCounter` postponed to C4 (§4).
  Rule → test:
  - Lossless round trip: `test_adapter_anthropic.py::test_every_supported_conversation_round_trips`,
    `test_adapter_openai.py::test_every_supported_conversation_round_trips` (hypothesis), plus every named case
    of §3.2 (string and a single text block, `cache_control`, the three kinds of tool_result content and
    `is_error`, a malformed `arguments`, `content: null` / `""` / no key, several tool messages of a parallel
    batch, `name` and unknown keys, the position of system / developer).
  - Input outside the table is refused: both adapters' `test_input_outside_the_supported_range_is_refused`.
  - `Opaque` comes back unchanged:
    `test_blocks_the_core_does_not_handle_become_opaque_and_come_back_unchanged` (the same object).
  - Estimate by kind, never count `raw`: the five tests in `test_count.py` for image / thinking / text document
    / PDF / unknown block.
  - Zero-dependency core: `test_core_imports.py` (imports per §2's list in a subprocess and checks
    `sys.modules`).
  - `fixed` over budget: `test_fixed_content_over_the_budget_does_not_fit_before_any_cutting`.
  - Tests do not touch the network: `test_no_network.py`.
  Seen red: the new tests all failed first on an empty package; then the code under test was broken one at a
  time to confirm the key tests turn red (the core importing one more third-party module, turning off the
  socket-refusing fixture, Anthropic not recording the list form, the PDF page count always 0, OpenAI parsing
  `arguments` early), and restored.
  `/simplify` (subagents from four angles): accepted — the counter no longer reads the vendors' block structure,
  and the adapters take out `Opaque.text` / `Opaque.pages` during conversion (a PDF is decoded only once); the
  content form changed from a prefixed conventional key in `extra` to a typed `content_form`, `is_error` became
  `bool | None`, `tool_calls: []` and the extra fields of `function` stay in `extra` as they are, and the
  namespace keys were removed; the duplication between the two adapters (plain-text check, text block, opaque,
  string or list) moved into `adapters/_common.py`; added `fixed_texts()`; `Opaque.estimated` removed
  (`tokens is None` means estimated); OpenAI output keeps the order when content is mixed; the fields in
  `Report` that only C1 uses and `Action` were removed, to be added back in C1. Skipped — caching the counting
  result (C1's `fit()` has to accumulate per message rather than recount everything after each deletion; noted
  here for C1); a JSON round trip of the `tool_use` input every time (later, when a host has large inputs). The
  new tests (measures, order, `tool_calls: []`, `fixed_texts`, `tokens` not asking the counter) were all seen red
  by breaking the code under test.
  Code review: Codex was unavailable, replaced by an adversarial subagent review. All 6 findings held and were
  all fixed, with regression tests red then green:
  1. An Anthropic `tool_use.input` with non-string keys, tuples or NaN was accepted but could not come back →
     `allow_nan=False` and check `loads(dumps(x)) == x`, and if not equal `UnsupportedInput`
     (`test_tool_use_input_that_would_not_come_back_equal_is_refused`).
  2. An OpenAI tool call with no `type` had a `type` added on output → accept only `type: "function"`
     (`test_a_tool_call_that_is_not_a_function_call_is_refused`).
  3. Output and input share nested objects → written into §3.1 as a convention (deliberately not deep-copied).
  4. The SDK's `tool_calls: null` was rejected → accepted and kept as it is
     (`test_the_sdk_shape_with_tool_calls_none_survives`).
  5. The property test's generator missed some named cases → added (floats, extra fields on a message and on a
     tool_use, document / redacted / unknown blocks, the four forms of `tool_calls`, extra fields of `function`,
     `name`, images and files in tool content, `refusal`); additionally run once with 3000 examples each, no
     failures.
  6. `CharEstimate` and `Budget` accepted impossible numbers → validated at construction and in `available()`
     (two parametrised tests).
  Recorded but not changed: `restore` lets the core field win when a hand-built `extra` has the same name as a
  core field (an `extra` the adapters produce never does); `Message` cannot be hashed because `extra` is a dict
  (to be revisited if C1 needs it); the limits of the PDF page-count estimate (§4).

- **C1a (2026-11-26)**: `structure.py` (batches, turns, pairing check), `steps.py` (`Entry` / `Unit` /
  `StepContext`, the `AlwaysStep` / `CuttingStep` protocols, the four steps, fixed placeholder text), `state.py`
  (`ContextState`, `Cut`, JSON with a version), `fit.py` (the flow, high and low water marks, replaying and
  advancing cut points, the runtime check after each step), `testing.py` (`check_step`); new errors
  `BrokenPairing`, `StaleState`, `UnreportedChange`, `ProtectedChange`; `Report` gains actions and before / after
  numbers. Deviations from this document: the batch rule was tightened (while a call is unanswered, the next
  message must be one carrying results); `DropOldest`'s placeholder goes at the position of the first deleted
  message; no step touches a protected message.
  Rule → test (`tests/test_fit.py` unless noted):
  - §5.2-1 pure function, input unchanged:
    `test_the_input_is_untouched_and_the_same_input_gives_the_same_output`.
  - §5.2-2 batches are not split: `test_mask_keeps_the_last_n_batches_counting_a_parallel_batch_once`,
    `test_an_unfinished_batch_is_never_masked`; pairing check `test_structure.py::test_broken_pairing_is_refused`
    (five cases); a step that splits a pairing is caught
    `test_a_step_that_breaks_a_rule_is_caught_at_run_time[break-pair]`.
  - §5.2-3 pinned: `test_a_pinned_batch_is_skipped_and_the_next_one_masked`,
    `test_drop_oldest_never_drops_a_turn_holding_a_pinned_message`, `...caught_at_run_time[touch-pinned]`.
  - §5.2-4 the human message of the last turn is protected, the newest tool result can be truncated:
    `test_mask_opaque_removes_old_images_but_not_the_last_human_turns_or_thinking`,
    `test_the_newest_tool_result_is_truncated_rather_than_failing_every_call`.
  - §5.2-5 report and runtime check: `test_cutting_steps_stop_at_the_low_water_mark` (actions compared one by
    one), `...caught_at_run_time[silent-edit]`, `test_testing.py::test_a_host_step_that_hides_a_change_fails`.
  - §5.2-6 fixed placeholder: the four tests for truncate, mask, opaque and drop all compare the constants byte
    for byte.
  - §5.2-7 thinking: `test_a_step_that_removes_a_thinking_block_is_caught` and the opaque one.
  - §5.1 high and low water marks: `test_nothing_is_cut_below_the_high_water_mark`,
    `test_cutting_steps_stop_at_the_low_water_mark`,
    `test_when_every_step_has_run_and_it_still_does_not_fit_it_raises_with_the_report`,
    `test_compaction_is_due_when_the_cut_history_is_still_over_the_high_water_mark`.
  - §5.4 state: `test_with_state_old_messages_change_only_on_the_turns_that_cross_the_high_water_mark` (20
    turns, cache stability), `test_state_is_replayed_even_when_the_history_would_fit_without_it`,
    `test_state_from_a_history_that_changed_before_the_cut_is_stale`, `test_without_keys_no_cuts_are_kept`,
    `test_state_survives_json_and_an_unknown_version_is_refused`.
  - `check_step`: `test_testing.py` (the four built-in steps pass and really change something, a host step that
    hides a change fails).
  Seen red: the tests were written first and all failed with no implementation; then 14 pieces of key code were
  broken one by one (the high-water threshold, the low-water mark, replay, the protection check, the unreported
  check, the thinking check, the pairing check, the digest, the unfinished batch, opaque protection, drop
  protection, keep_last, an unanswered call, the binary search), a test turned red at each, and the code was
  restored.
  `/simplify` (two subagents): accepted — `fit()`'s cutting loop used to replay and recount the whole history for
  every unit added (356–526 ms on the first call with 1000 messages); it now counts each message once and
  afterwards recounts only the messages a step reported, and uses a binary search to fix the number of units (6–21
  ms for the same input, the number of actions unchanged); `MaskToolResults` takes the tool name directly from
  the batch's call message instead of rebuilding batches every time; the three steps share `_rewrite`; the
  duplicate-key check shares `check_unique`; the tests' `COUNTER` uses the one in `lite_context.testing`; dead
  branches and one type ignore removed; the thinking check looks only at the messages that were touched.
  Decided in the design: a new compaction carries no cut points, and cut points measure the history after the
  summary (§5.4); applying the summary is an internal step that goes through the same runtime check (§6.2), and
  C1b implements it that way. Skipped — making `Unit` a generic payload (no need once the tool name is taken from
  the message).
  Code review (Codex unavailable, subagent in its place): 9 findings, 8 fixed and 1 recorded; regression tests
  red then green:
  1. After one cutting step advanced, a later step's unit list got shorter, "first n units + digest" no longer
     matched, and normal growth raised `StaleState` → the cut point became "which message it cut through + a
     digest of the input refs up to there", replayed by position (§5.4)
     (`test_two_cutting_steps_keep_working_as_the_history_grows`, which requires "one step crossing another
     step's cut point" to occur).
  2. When protection left an old image it was inserted into the middle of the unit list, with the same false
     alarm → the same fix (`test_protection_moving_off_an_old_image_does_not_make_the_state_stale`).
  3. A caller's key `#1` collided with the position ref of a message with no key, the count was wrong, and going
     over budget raised no error → the `#` and `+` prefixes are reserved; the check also requires refs to be
     unique (`test_keys_that_look_like_internal_refs_are_refused`, `...caught_at_run_time[duplicate]`).
  4. `DropOldest`'s placeholder was longer than the messages it deleted, so something that fit raised
     `DoesNotFit` → a cut point that makes the total larger is not adopted
     (`test_a_cut_that_would_cost_more_than_it_saves_is_not_taken`).
  5. `check_step` did not test state and growth → added a growth run and a run without keys; `fit()` sorts the
     units by position itself, so "listing units from new to old" is no longer an error
     (`test_the_order_a_step_lists_its_units_in_does_not_matter`); the growth check catches a step whose output
     drifts as the history grows (`test_a_step_whose_output_drifts_as_the_history_grows_fails`).
  6. Truncation went by piece rather than by whole result → by whole result
     (`test_truncation_limits_a_whole_result_not_each_text_part`).
  7. `head_tail` overlapped head and tail under a non-additive counter → the tail is taken only from what is left
     (`test_head_and_tail_never_overlap_even_with_a_counter_that_is_not_additive`).
  8. Recorded only: `DropOldest`'s placeholder is a user message with text, so it is taken as the start of a
     turn; in `fit()` only `DropOldest` itself cuts by turn, and it looks at its own input, so it has no effect
     for now.
  9. `from_json` raised various exceptions on malformed input → unified as `ValueError`
     (`test_a_malformed_state_is_a_value_error`).
  Also: every candidate tried during the binary search must pass the runtime check (otherwise a rule-breaking
  output that was given up would go unnoticed).
  The final version of test 1 (budget 1200, 15 turns) was not run against the old code; its first version
  (budget 600) and test 2 were red on the old code. Timing after the fix is the same order of magnitude as before
  the fix (1000 messages, three cutting steps, first call 27 ms).
  Second round of review (the cut-point change was large, so it was reviewed again under the close-out rule; the
  subagent wrote a random-growth probe: a random history grown one message at a time, random step combinations
  and budgets, passing state back each time): all 6 findings fixed, with regression tests red then green:
  1. A cut point records a whole message, but `MaskOpaque` had one unit per image and might cut only part of one
     message, and the next replay cut the rest too → `fit()` stops only at message boundaries
     (`test_a_host_steps_units_in_one_message_are_cut_together`, a host's own per-image step).
  2. `MaskOpaque`'s `keep_last` boundary fell in the middle of a message, the same problem → the unit became one
     message (`test_images_in_one_message_are_cut_together_so_replay_cuts_no_more`).
  3. The binary search assumed "the more cut, the more saved", and masking a result that is shorter than the
     placeholder makes it longer, so it missed a cut that does exist → changed to the estimate plus verification
     described above (`test_a_cut_is_found_even_when_masking_some_results_makes_them_longer`,
     `test_when_no_cut_reaches_the_low_mark_the_cheapest_one_that_fits_is_taken`). With a linear scan the first
     call fell back to 392–667 ms; with estimate plus verification it is 10–15 ms (1000 messages).
  4. When the same `Text` object appeared twice the truncation result was duplicated → look up by position
     (`test_truncation_handles_the_same_text_object_twice`).
  5. Truncation with the marker added came out longer → do not truncate
     (`test_truncation_never_makes_a_result_longer`); the inputs of the two original truncation tests were
     therefore made longer.
  6. `check_step`'s samples had no message with several images → added, grown one message at a time, three
     budgets.
  7. Recorded only: an image that leaves protection is cut at the next replay (accepted in §5.4); `check_step`
     and the random-growth test let this one pass.
  The random-growth probe went into the repo: `tests/test_growth.py` (150 seeds; each checks no exception, the
  same input gives the same output, and the prefix is unchanged when the cut points did not move); another 1000
  seeds were run by hand, no failures.
  Seen-red addendum: message boundaries, the cheapest-cut fallback, unit ordering, unique refs and not adopting a
  more expensive cut point were each confirmed by breaking the code and seeing a test turn red.

- **C1b (2026-11-26)**: `compaction.py` (`SummaryPolicy`, `plan_compaction`, `CompactionPlan.accept`,
  `acompact`, `Summary`, `covered_end`); `Compaction` moved into `state.py`, and `ContextState.compaction` is
  stored and read back with the JSON; `fit()` takes applying the summary as an internal step that goes through
  the same runtime check; new errors `EmptySummary`, `TruncatedSummary`. Deviations from this document:
  `created_at` is an ISO string; `covered_count` removed (the digest already covers the count); `plan.covered`
  gives the covered keys.
  Rule → test (`tests/test_compaction.py` unless noted):
  - Covered range: `test_a_plan_covers_every_turn_but_the_kept_ones_and_renders_them`,
    `test_a_pinned_prefix_stays_out_and_a_later_pinned_message_stops_the_cut`,
    `test_nothing_to_cover_gives_no_plan`, `test_compaction_needs_keys`.
  - A later one takes in the earlier one: `test_a_later_compaction_takes_in_the_earlier_summary`,
    `test_nothing_new_to_cover_after_a_compaction_gives_no_plan`.
  - A truncated or empty summary fails, copied header lines are removed:
    `test_an_empty_or_truncated_summary_is_refused`, `test_header_lines_the_model_copied_are_removed`.
  - Applying the summary, the report, input unchanged:
    `test_fit_shows_the_summary_in_place_of_what_it_covers_and_reports_it`.
  - Stale: `test_a_summary_that_no_longer_matches_the_history_is_stale` (deleting the last covered message,
    inserting before, changing a key, not on a turn boundary), `test_a_summary_with_keyless_messages_is_stale`.
  - Cut points measure the history after the summary, a new summary carries no cut points:
    `test_cuts_are_measured_after_the_summary_and_start_again_after_a_new_one`.
  - JSON: `test_state_with_a_compaction_survives_json`; `acompact`:
    `test_acompact_plans_calls_the_callers_model_and_accepts`.
  - End to end: `tests/test_growth.py::test_a_growing_history_with_summaries_replays_stably` (60 seeds, a summary
    written whenever needed, 60 summaries in all; no exception, and the prefix is unchanged when neither the cut
    points nor the summary moved).
  Seen red: the tests were written first and failed at collection with no implementation; then 10 places were
  broken (the digest, the turn boundary, truncation, header cleanup, stopping at pinned, nothing new, taking in
  the earlier one, `state` keeping the summary, the no-key check, the count check): for the first 9 a test turned
  red; breaking the count check turned no test red, because the digest already covers the count, so the count
  field was deleted; the no-key check originally had no test, one was added and seen red.

  Close-out: `/simplify` (one subagent, four angles) and the code review (Codex unavailable, subagent in its
  place) ran at the same time.
  Review, 7 findings: 1, 2, 5 `end` could land at the end of the history (`keep_last_turns=0`, no human
  messages, a negative value), the summary covered the current turn or an unfinished batch, and the resulting
  state was unusable → `end` is always the start of some turn, and `keep_last_turns >= 1`
  (`test_keep_last_turns_must_keep_at_least_the_current_turn`,
  `test_a_history_without_human_turns_gives_no_plan`, `test_fewer_turns_than_kept_gives_no_plan`,
  `test_the_plan_never_covers_an_unfinished_batch_or_the_last_turn`); 3 a `provenance` that is not JSON failed
  only at storage time, and a tuple was not equal after a round trip → normalised as JSON in `accept`
  (`test_provenance_is_stored_as_json_gives_it_back`); 4 header cleanup recognised only a hexadecimal mark → the
  plan records the mark and cleans only that one (`test_copied_headers_are_removed_whatever_the_mark`); 7 the
  previous summary was given to the model with its preamble → remove it first
  (`test_the_earlier_summary_goes_back_to_the_model_without_its_preamble`); 6 consecutive user messages after the
  summary → written into §6.2.
  `/simplify`, 8 findings: accepted — the two identical digest functions merged into `digest`; `pinned_prefix`
  moved to `structure.py`; `next()` instead of a list of which only the first item is taken; `render` split out
  into `_render` per message; `_summarised` takes only `entries`, and `SUMMARY_REF` goes at the top of the
  module; `Compaction`'s JSON uses `asdict` / `fields`; `__all__` lists only what the module defines. Skipped —
  `human_turns` computed twice (O(n), cheap); making `SummaryPolicy` a Protocol (lite-agent can inherit from it
  or `replace` it in C7, to be decided then).
  Seen red: regression tests red then green; the fix sites were broken one by one (the `keep_last_turns`
  validation, too few turns, JSON normalisation, cleanup by mark, removing the preamble), a test turned red at
  each; for "too few turns" no test turned red at first, and `test_fewer_turns_than_kept_gives_no_plan` was
  added.

- **C2 (2026-11-26)**: `docs.py` (`split`, `pack`, `halving`, `reduce_tree`, `select`, generic);
  `steps._longest` moved to `count.longest_prefix` (`split` uses it too), which fixes the window from the text's
  own character / token ratio and then binary-searches, about 20 ms to split 1 MB of text into 8k pieces.
  Deviations from this document: one module rather than a `docs/` package; `pack` does not cut an oversized unit
  (the unit's type is up to the caller and cannot be cut), and over the limit it is `DoesNotFit`; `reduce_tree`
  returns the notes that fit in the budget together, and the final reduce is done by the caller; `select` stops
  at the first one that does not fit.
  Rule → test (`tests/test_docs.py`): fits means one piece as given
  `test_text_that_fits_is_one_piece_exactly_as_given`; coverage, overlap, cutting at newlines
  `test_pieces_cover_the_text_overlap_and_end_at_a_line_break`; it can still be cut with no newline
  `test_a_long_line_without_breaks_is_still_cut_to_size`; heading
  `test_each_piece_carries_the_heading_above_where_it_starts`; not over the limit under the default estimate, and
  the speed on 1 MB `test_split_respects_the_limit_with_the_default_estimate_and_cuts_a_large_text_quickly`;
  two for `pack`; three for `halving` (halve only what is too large, any other failure is raised as it is after
  one call, still raised at the minimum); five for `reduce_tree` (no merge when it fits, merge by level and in
  order, failure when a single note cannot be merged, failure at the depth limit, failure when a merge result is
  over budget); one for `select` (order, de-duplication, stop at the first one over).
  Seen red: the tests were written first and failed at collection with no implementation; then 13 places were
  broken, and at first 4 had no test turn red: the early return when it fits (equivalent to the general path,
  deleted), cutting at newlines (the test text happened to put whole lines at exactly 500, so the line length was
  changed), halving only what is too large (not halving would also end up raising the same exception, so the
  test was changed to assert only one call), the depth limit (the test hit "no progress" first, and a scenario
  that makes progress at every level was added); after the changes all turned red.

  Close-out: `/simplify` (subagent, four angles) and the code review (Codex unavailable, subagent in its place)
  ran at the same time, and after de-duplicating the two:
  1. `halving` with `min_size=0` recursed forever → at least 1
     (`test_halving_needs_a_minimum_of_at_least_one`).
  2. Catching `BaseException` handed cancellation to the caller's predicate → catch only `Exception`
     (`test_a_cancellation_never_reaches_the_size_predicate`).
  3. After `select` stopped it would lose chunks and `over_budget` contained duplicates → keep de-duplicating
     after it stops, and the three lists partition the input
     (`test_select_sorts_every_chunk_into_exactly_one_list`).
  4. `split` counted all the remaining text for every piece, 66 times the full text under a real tokenizer →
     count each small segment once, about 3 times
     (`test_split_counts_a_bounded_window_not_the_whole_rest_each_time`); there is a test for a non-additive
     counter (`test_a_counter_that_charges_more_for_the_whole_than_its_parts_is_still_respected`); a
     self-contradictory counter gives an error instead of an endless loop.
  5. A piece that starts on a heading line carried the previous heading → it carries the heading of its own
     section (`test_a_piece_that_starts_on_a_heading_carries_that_heading`; the expectation of the original
     heading test changed with it).
  6. `Halved` could not say which results came from which piece → one record per call
     (`test_halving_says_which_results_each_call_gave`).
  7. The four identical `DoesNotFit` constructions merged into `_too_big`; `reduce_tree` counts each note once
     per level; dead code removed. In the `Report` that `DoesNotFit` carries, only limit and overhead are
     meaningful here — not made optional (that would change the public interface), recorded here.
  Seen red: regression tests red then green; the fix sites were broken one by one and a test turned red each
  time; after breaking "cut an over-long small segment first" the test hung, a guard was added so that it
  reports an error, and after that the break shows as a failing test rather than a hang.

- **C3 (2026-11-26, preparation)**: version 0.1.0; CI adds `package` (`uv build`, installs only this wheel in a
  clean 3.11 environment, and runs `scripts/smoke.py` with `python -I`: the README's full example, adapter →
  `fit` → JSON state → `acompact` → `fit` again → `split`, asserting concrete results) and `release` (on a v*
  tag, publishes the one `package` built to a GitHub Release). Local verification: a 3.11 environment made by
  uv, a 3.11 venv made by pip, and a new uv 3.12 project with `uv add <wheel>`, and the smoke run passed in all
  three. The README was changed to the real usage and installation. Found: when the step list is first assigned
  to a variable, mypy infers `list[object]`, and it must be annotated `list[Step]` (the README says so).
  Release (owner agreed): `v0.1.0` tag → all four CI jobs green (test 3.11, 3.12, package, release) → a GitHub
  Release with `lite_context-0.1.0-py3-none-any.whl` and the sdist. Acceptance: installed from the release URL
  with pip (3.11 venv) and `uv add` (3.12 new project), and the smoke run passed in both.

- **v0.1.1 (2026-11-26, found while preparing C4)**: the wheel had no `py.typed`, so a host's pyright / mypy
  treated lite-context as an untyped package. Added `src/lite_context/py.typed`; CI's `package` job checks that
  the wheel contains it (before adding it, the check was confirmed to fail on a local build).

- **v0.1.2 (2026-11-26, found in the review of host A's adoption)**: some OpenAI-compatible backends number
  the call `call_0` in every reply, and `MaskToolResults` looked up results by call id across all messages, so
  when an old batch was masked, the newest result that reused the same id was masked too, and the model did not
  get the result it had just asked for. Changed so that each batch masks only the results in its own result
  messages, and the tool name also comes from its own call message
  (`test_a_call_id_reused_by_a_later_batch_does_not_mask_the_later_result`, red before the change); the
  random-growth test also gained histories that reuse ids (it checks prefix stability and cannot catch this
  error, so the unit test is responsible).

- **C4 host A (2026-11-26)**: host A's agent tool loop (inside a Temporal workflow) uses `fit()` before each
  model call (`MaskToolResults(keep_last=2)`, budget 150,000, the system prompt and tool definitions counted as
  `fixed`) and passes the `ContextState` along step by step in workflow memory; when it does not fit, the run
  ends in failure. The host wrote an adapter of about 50 lines itself (its message types ↔ neutral types,
  lossless round trip). The case flow (grouping by page, halving once) was not switched: its semantics differ
  from `pack` / `halving` (§15).
  Measured (synthetic data: host A allows only synthetic data; 20 steps, each tool result at the 20,000-character
  cap, the real system prompt and tool definitions):
  | Budget | Estimated tokens at step 20 without masking | Steps over budget without masking | Largest send after masking | Steps where the cut point moved | Failure |
  | --- | --- | --- | --- | --- | --- |
  | 150,000 (default) | 110,013 | 0 | 110,013 (nothing masked) | 0 | no |
  | 60,000 | 110,013 | 9 / 20 | 53,039 | 4 / 20 | no |
  That is, behaviour is unchanged at the default budget; with a small window the 9 steps that would have failed
  can all be sent, and the prompt cache is invalidated in only 4 steps.
  Found during adoption and fixed in the library: the wheel had no `py.typed` (v0.1.1); the newest result was
  masked when a call id was reused (v0.1.2). Also learned on the host side: any lite-context exception must end
  the run rather than let the workflow task retry forever; `fit`'s `DoesNotFit` message is clear enough to give
  straight to the user (the case of `fixed` over budget also states its reason).
  Pass condition: host A's `make check` all green (backend 347, frontend 33 + 51), window-overflow failures are 0
  in the scenarios of the table above, and input tokens are no higher than before adoption (equal at the default
  budget).
- **C4 host B (2026-11-26, trialled then reverted, the owner decided not to change it for now)**: below is what
  the trial did and its measurements; host B's code has been restored to its original state, and the patches did
  not go into the host repository. Host B's chat endpoint (the browser sends the whole conversation every turn,
  as Anthropic-format dicts) calls `fit()` before the model call: `MaskOpaque(keep_last=1)` then
  `DropOldest(keep_last_turns=4)`; the budget is the smaller of the model window and the vendor's per-minute
  quota, minus the reply's `max_tokens`; the system prompt and the retrieval excerpts count as `fixed`. Stateless
  (owner's decision): no `ContextState` is stored and it is recomputed every turn, so after 90% the cut point may
  move every turn and the prompt cache is invalidated with it. It used the built-in Anthropic adapter, so the
  host wrote no adapter; when something is trimmed the response carries counts, and when it does not fit it
  returns 413 and does not call the model. All of host B's tests passed at that time (backend 822, frontend 320).
  Measured (synthetic data: 30 turns, about 2,000 characters from each side per turn, a 20-page PDF on each of
  turns 0/6/12/18/24, turn 30 at about 234k estimated tokens):
  | Window | Turns over budget without trimming | Largest send after trimming | Turn 30 | Failure |
  | --- | --- | --- | --- | --- |
  | 1,000,000 (the chat model host B configures) | 0 / 30 | 234,014 (nothing trimmed) | unchanged | 0 |
  | 256,000 | 0 / 30 | 187,096 | 2 files replaced by placeholders | 0 |
  | 60,000 (forced) | 24 / 30 | 48,658 | 4 files, 50 messages | 0 |
  Learned in the trial (all on the host side, the library needs no change; do the same when it is adopted again):
  1. The estimate must follow the form the host **actually sends**. For most OpenAI-protocol vendors host B turns
     a PDF into text before sending (at most 400,000 characters over all PDFs together), and estimating by page
     count would judge a 70-page PDF that used to be sendable as a 413. The host uses a `CharEstimate` subclass
     to cap PDFs.
  2. The budget must look at the vendor's **per-minute quota**, not only the window: on an 8,000 TPM tier a
     request over the quota is simply rejected. At first the host copied the "keep a fifth in reserve" margin
     from its document reading, which blocked long replies and long system prompts that used to be answerable
     (found in the second round of review). A chat turn is only one request, so it was changed to use the full
     quota.
  3. `DoesNotFit` has two causes: the kept turns do not fit, or `fixed` itself is over. The message to the user
     has to tell them apart and cannot say "the latest one is too large" for both.

- **C5 (2026-11-26)**: `Transcript` (async), `InMemoryStore`, `PgStore` (extra `pg`), the message JSON codec,
  `litecontext migrate` / `pending`, version 0.2.0. Acceptance: the same set of store contract tests runs on both
  memory and PG (read back in order, nothing written when `expected_seq` is wrong, a batch's seqs must follow on,
  an unknown transcript, 8 concurrent appends with exactly one succeeding); on PG UPDATE / DELETE / TRUNCATE are
  rejected by the trigger; for a scripted conversation that grows, masks and writes one summary, the projection
  of memory and PG is byte for byte the same at every step; another process (`python -I`) reads back from PG and
  gets the same projection. The core still loads no third-party module (the wheel installed in a clean 3.11
  venv, smoke passed, `psycopg` was not imported).
  Mutation checking found two places the tests did not bite, and they were added: with PG's `FOR UPDATE` removed
  the concurrency test still passed (the test store's connection pool started with only 1 connection, so the
  concurrent appends were in fact queued; after changing it to start with 4 connections, removing the lock gives
  `UniqueViolation` and the test fails); with the projection using the first state rather than the latest one
  the tests still passed (only one state had ever been stored; a case that stores a mask state first and then a
  summary was added).
  Close-out: `/simplify` from four angles + review (Codex quota used up until 2026-12-18, a Claude subagent in its
  place). Review found and fixed (a failing test was written first each time): the codec did not check at write
  time the types it checks at read time, so a message like `pinned=1` could be written and then the whole
  transcript could not be read; NUL / a lone surrogate was accepted by the in-memory store and made PG raise
  psycopg's raw exception; after a first connection failure the store was unusable for good (a closed connection
  pool cannot be reopened); a database error after connecting was not a `StoreError`; jsonb reordering keys.
  Simplify accepted: the storage layer became its own `store.py` (pg no longer imports fit through transcript),
  `check_batch` also checks the tail, the schema check moved into `migrate` / `pending`, state encoding moved into
  the codec, the codec's allowed values are taken from `types`' Literal, tests share one `keyless_turn` and
  `fresh_schema`, and the network guard lets the database host through only in tests that asked for `pg_url`.
  Not accepted: a large PDF being serialised about 3 times per append, and `project()` reading the whole log every
  time (no host stores large files in a transcript now; later use a type walk in place of dumps/loads and an
  incremental read by `after`).

- **C6 (2026-11-26)**: the REST shell `litecontext serve` (extra `server`, §9), version 0.3.0. Two functions were
  added to the library for it: `Transcript.plan_compaction()` (the summary's plan and the tail seq at that time;
  `compact()` was changed to use it too) and `compaction.accept_summary()` (accepts a summary from the parts of a
  plan; `CompactionPlan.accept` delegates to it), so that the shell has no summary logic. Acceptance: every
  endpoint's output equals the library's on the same input field by field (fit in the three formats, summary plan
  / accept, split / pack / select, and the transcript's whole script were compared on both memory and PG); it can
  bind only to a loopback address; a request body over the limit, declared or actual, gets 413 (when the declared
  size is over, not a byte is read). Started by hand once (8750 confirmed free first, shut down afterwards):
  `/v1/fit` masked the old tool results, 145 → 49 estimated tokens; a foreign Host got 400.
  Close-out: `/simplify` + review (Codex quota used up, a Claude subagent). Review found and fixed (a failing test
  first): `heading` that is not a valid regular expression gave 500; joining the request body block by block was
  quadratic (30 MB in 4 KB blocks took 9.5 seconds and blocked the event loop; changed to a bytearray); a bad
  port or body limit in the CLI threw a traceback (and a bad port even stopped `pending` from starting); the
  library's OpenAI adapter let through a system message whose content was not text, counting it as empty text
  (now `UnsupportedInput`). Simplify accepted: one global error handler in place of a wrapper on each route, the
  summary-policy fields merged into one mixin, state encoding going through the codec, `accept_summary` moved
  into the library, `entries(after=)` left to the store to filter. Mutation checking: all 16 places were caught
  by tests; a test defect was found along the way: when `check_host` was broken, the `serve` test really started
  uvicorn, and now the CLI tests always replace `uvicorn.run` with a stub that fails first. Not accepted: making
  the error code an attribute of the exception class (it is now derived from the class name, so a rename would
  change it — decide when there is a second shell); the `heading` regular expression's ReDoS is covered only by
  the loopback address and the 200-character cap (Python `re` has no timeout).
  Another round of review after the fixes: no new bugs; two small problems fixed — a transcript id with a NUL was
  a 503 on PG (now `Transcript` itself checks the id by the same rule as the codec, 1–200 characters, and every
  store gives 422); the error for the request-body limit mentioned a `--max_body` parameter that does not exist.
  Accepted trade-off: every `ValueError` in the library is returned as 422, so a bug in the library itself that
  appears as a `ValueError` would also be 422 rather than 500.

- **C7 (2026-11-26)**: lite-agent v0.2.5's compactor switched to lite-context 0.3.1 (done in the lite-agent
  repo; the owner decided: lite-context is its core dependency, and the scope goes only as far as compaction).
  The prompt (`ASK` / `END`), the summary frame (`SUMMARY`) and the transcript rendering (the newly public
  `render_transcript`) come from here; lite-agent keeps its own cut point (by `model_turn`, on event seqs, while
  `plan_compaction` goes by human turn, on messages with keys, so does not apply) and its rule for removing
  copied-back headers (any hexadecimal header). So the C7 row's "it provides its own `SummaryPolicy`" turned out
  not to be needed: it does not call `plan_compaction` / `accept` and uses only the rendering and the frame.
  Acceptance: the sha256 of lite-agent's 38 existing test files is the same before and after, and all 503 tests
  pass without changing a line; the golden outputs generated with the v0.2.4 code are byte for byte the same;
  the review compared the old and new implementations over 20,000 random histories, with no difference on the
  paths a loop can reach. fit() / masking was not wired into lite-agent (the scope the owner chose). C0–C7 are
  all done.

## 15. Open questions

- **A vendor's count_tokens**: for Claude host B uses the SDK's `messages.count_tokens` (a network call). The
  core's `Counter` is synchronous; whether to add an async counter, or to let the host compute first and pass
  `Opaque.tokens` / the budget, was to be decided when C4 was adopted.
  C4 conclusion (host B): the estimate is good enough for chat; document split / halving / tree reduce were
  therefore not switched, and host B could switch them over only if the cost of `split` / `halving` /
  `reduce_tree` could be async (or accepted a precomputed number).
  v0.3.2 conclusion: use calibration, no async counter (§4) — host B's own document tools already "count a whole
  document once and work out the ratio", so do it that way when it is switched over (host B is deferred, the
  owner's decision).
- ~~**A PDF whose pages cannot be counted**~~ (C4 host B): solved in v0.3.2 — inflate the compressed object
  streams and count again, then look at the page tree's `/Count` (§4); no guessing from the byte size.
- **Summarising the report**: both hosts counted "how many masked / how many dropped" themselves by
  `Action.what`. If a third host needs it too, add a summary method to `Report`.
- **How to measure summary quality**: which of masking and summaries is better is a reported public result. After
  C4, compare them with lite-eval on a host's real tasks, and before measuring, this document does not recommend
  either one as the default.
  First measurement (2026-11-26, lite-agent's long-task eval; synthetic tasks, qwen3:8b, 3 runs per task per
  configuration): with masking before summaries, 32% fewer tokens and 57% less time were spent, but the model
  answered using only the last one or two results still visible and made up the ones it could not see (the true
  weight of a delayed shipment answered correctly 9/45, with summaries only 22/45; the first report's weight 0/3
  correct against 3/3); the placeholder told it to re-read, and in 9 runs it did so 0 times. lite-agent therefore
  does not turn masking on by default. This is still not "a host's real tasks", so this item stays open; the next
  step is the placeholder itself (`MASKED_RESULT` is a fixed constant, §5.2 rule 6) or a stronger model. Record:
  lite-agent design §15 / §16 (github.com/ArthurGao/lite-agent, `docs/lite-agent-design.md`), KB
  `global/llm-evals/evals/2026-11-26-lite-agent-masking-long.md`.
- **Server-side summaries**: in REST mode the service calls a lite-model role itself to write the summary
  ("a later option" in §0). To be done when a non-Python host wants it.
- **Token-level compression** (LLMLingua and the like): needs a small local model, suits long documents rather
  than history; when a host wants it, it is added as an optional step of §7.
- **The Python 3.11 floor**: once host B moves up to 3.12, whether to raise it to 3.12 like the other lite
  modules.
- **Where the project KB lives**: `KB/projects/lite-context/` (created in S0).
- **Whether `pack` should allow a single over-budget unit to form a group of its own** (C4 host A): host A's
  page grouping lets a page over budget form its own group and hands it to the model (and halves once more if the
  model rejects it), whereas `pack` goes straight to `DoesNotFit`. Host A's case flow is not changed this time
  (its §8.3 is a deliberate design); an option is added when a host needs it.
