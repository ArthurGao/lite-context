# lite-context 设计文档

**状态：v0.1.1（2026-11-26：wheel 带 `py.typed`）；v0.1.0 已发布（C3，2026-11-26，GitHub Release wheel）。C2 完成：长文档工具（`split` / `pack` / `halving` / `reduce_tree` / `select`）；C1 完成：C1b 摘要（`plan_compaction` / `accept` / `acompact`、`SummaryPolicy`、`fit()` 应用摘要）；C1a `fit()`、四个步骤、`ContextState` 与切点、运行时检查；C0 完成：中立类型、Anthropic / OpenAI adapter、计数与预算。仓库 github.com/ArthurGao/lite-context。第一批宿主：两个自带
LLM 层的 Python 项目（下称宿主 A、宿主 B，§12），在 C4 同时接入。**
与代码冲突时，在同一次改动里更新本文，并在 commit 里说明原因。

## 0. 决策记录

决策的理由只写在这里，其它章节引用（§0）。"owner" 是主人定的；"提议" 是起草时的判断，主人没否决前按它做，
否决时在这里改。

| 日期 | 决策 | 谁 | 原因 |
| --- | --- | --- | --- |
| 2026-11-26 | 单独做一个库 lite-context，只管"这一次调用，模型该看到什么"；直接调用 LLM 的项目和 lite-agent 都用它 | owner | 多数项目不用 lite-agent，直接调 LLM；它们的 chat 历史、tool 结果、长文档各自手写了一套（或没有），context rot 和超窗口直接报错是共同的缺口 |
| 2026-11-26 | 形态：library 为主；REST API 是它上面的薄壳，不放逻辑 | owner | 与 lite-rag / lite-mem / lite-agent 一致；REST 给非 Python 的宿主 |
| 2026-11-26 | 显式调用，不做 lite-model hook 或"所有请求自动经过"的中间层 | 提议 | 在调用层悄悄改写请求，"模型看到的"就和宿主记下的对不上，回放、审计、fork 都会错（KB `llm-app-architecture` §9）。lite-cache 能做成 hook 是因为它不改请求内容 |
| 2026-11-26 | 核心不依赖任何 message 格式，也不依赖 lite-model：内部一套中立类型，边界上用 adapter | 提议 | 两个首批宿主都有自己的 LLM 层（一个用自有类型，一个用 Anthropic 格式的 dict），都不用 lite-model；强依赖 lite-model 等于要它们先换 LLM 层 |
| 2026-11-26 | 原始历史永不改写；压缩是一条记下覆盖范围的 `Compaction`，它和削减的切点一起放在 `ContextState` 里，由调用方保存（§5.4） | 提议 | KB §9.1 "compaction masks and keeps provenance"。调用方本来就存历史（数据库、workflow 状态），再存一条摘要记录即可，不必为此引入 lite-context 的存储 |
| 2026-11-26 | 装不下就报错（`DoesNotFit`），不悄悄丢内容 | 提议 | 两个首批宿主都明确选了"超限即失败"（一个 `case_too_large`，一个 `LLMInputTooLarge`），截断都当错误。lite-context 只做调用方声明过的裁剪 |
| 2026-11-26 | 需要模型的步骤（写摘要、map-reduce 的 merge）由调用方传入函数，lite-context 自己不调模型 | 提议 | 模型调用留在宿主自己的 role 层（workspace 规则：每次调用都经过 role、带 provenance）；核心因此是纯函数，可以直接在 Temporal workflow 里用 |
| 2026-11-26 | REST 模式下摘要默认由客户端写：服务返回写摘要用的 prompt，客户端调自己的模型后把摘要交回。服务端自己调 role 是以后的可选项 | 提议 | 服务不持有模型凭证、不花钱；模型选择和计费留在各个项目 |
| 2026-11-26 | Python ≥ 3.11，CI 跑 3.11 和 3.12 | owner | 宿主 B 跑 3.11；纯 Python 库支持 3.11 的成本很低 |
| 2026-11-26 | 核心零运行时依赖；lite-model adapter、tiktoken 计数器、PostgreSQL、FastAPI 各是一个 extra | 提议 | 宿主 B 用 pip requirements，依赖越少越容易接；核心是纯数据处理 |
| 2026-11-26 | 宿主 A、B 在 C4 同时接入 | owner | 一次验证两种接入形态：tool loop 的 masking，和 chat 历史的预算 + 摘要 |
| 2026-11-26 | 收尾流程与其它 lite 模块相同：`/simplify` → Codex 只读 review → 修复或写下否决理由 → 重跑全部检查 | owner | workspace 约定 |

## 1. 概述

lite-context 回答一个问题：**这一次调用，模型应该看到什么**。输入是调用方的完整历史（或一组文档、检索结果）
和一个 token 预算；输出是这一次要发送的内容，外加一份报告，写明裁掉了什么、压缩了什么、前后各多少 token。
发送仍由调用方自己做。

三类场景：

- **多轮历史**：chat、自写的 tool loop。历史越来越长，旧的 tool 结果占大头（§5、§6）。
- **长文档、检索结果**：一次调用装不下的文档要切分、分组、map-reduce；检索结果要在预算内挑选（§7）。
- **没有自己存储的宿主**：用 lite-context 的 `Transcript` 存历史（§8）。

### 来源

- KB `global/llm-app-architecture.md` §9（event log + 纯投影，压缩只遮盖）、§11（自写 loop 的截断）。
- lite-agent 的 S3b 压缩（它的设计 §6）：摘要事件、覆盖区间、"后一次压缩从头覆盖"、切点不拆开 tool call 和结果、
  截断或为空的摘要算失败。lite-context 把这些规则从 event log 里拿出来，变成不依赖 log 的形式。
- 两个首批宿主的现有代码（§12）：按 token 预算把页分组、超限对半重试、分层 reduce、tool 结果在产生时截断、
  "超限即失败"。
- 公开研究（reported，未在本项目测过）：Chroma "Context Rot"（2025）——输入越长质量越差，远没到窗口上限就开始；
  JetBrains "The Complexity Trap"（2025）——在 SWE-agent 上，把旧 observation 换成占位符的效果与 LLM 摘要相当、
  成本低得多。

### 目标

1. 一个 `fit()` 覆盖"历史太长"的常见做法，按调用方给的顺序逐级裁剪，每一步都写进报告（§5）。
2. 压缩可逆、可审计：原文不改，摘要是一条带覆盖范围的记录（§6）。
3. 长文档的切分、分组、对半重试、分层 reduce 做成共用工具，模型调用由调用方传入（§7）。
4. 不依赖任何 message 格式：Anthropic dict、OpenAI dict、lite-model 类型、宿主自有类型都能接（§3）。
5. 不联网、不花钱就能测：核心是纯函数，需要模型的地方用调用方传入的函数，测试里传 fake。

### 非目标（v1）

- 不调模型（例外：以后可选的服务端摘要，§9）。
- 不做 token 级压缩（LLMLingua 一类）；需要时作为 §7 的一个可选步骤再加（§15）。
- 不维护模型窗口表：预算由调用方算好传入（宿主有自己的表或配置）。
- 不做检索、embedding、rerank：那是 lite-rag。§7 的 `select` 只在已经排好序的结果里按预算挑。
- 不做长期记忆：那是 lite-mem。
- 不做 streaming。

## 2. 分层

```
宿主（Python，自有 LLM 层）──► adapter ──► lite-context 核心（纯函数）──► adapter ──► 宿主的 LLM 层 ──► 模型
宿主（Python，lite-model）─────► adapter(lite-model) ─┘
lite-agent（C7）──► 把事件映射成中立 Message ──► 同一个核心
非 Python 宿主 ──HTTP──► REST 壳（§9）──► 同一个核心
没有存储的宿主 ──► Transcript（§8）──► 同一个核心
```

依赖方向：核心只用标准库。核心 = `types`、`count`（`CharEstimate`）、`fit`、`steps`、`compaction`、`report`、
`errors`、`docs/`、`adapters/anthropic`、`adapters/openai`、`transcript/`（不含 `pg/`）、`testing`。
extra：`adapters/lite_model`（`lite-model`）、`TiktokenCounter`（`tiktoken`）、`transcript/pg/`（`pg`）、`server/`（`server`）；
核心不 import 它们。C0 的 `sys.modules` 检查就按这份清单。lite-context 不依赖任何 lite 模块。

## 3. 中立类型与 adapter

### 3.1 类型

全部是 frozen dataclass（3.11 可用的写法：`TypeAlias`、`TypeVar`，不用 PEP 695）。

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

- `extra` 放核心不关心、但转回厂商格式时必须原样带上的字段：`cache_control`、OpenAI 的 `name`、`tool_calls: []`、
  `function` 里多出来的字段。核心不读它，只搬运；被 mask 的部分不保留 `extra` 之外的原文。
- `content_form` 记下 content 原来的形式（字符串、列表、`null`、没有这个键），是有类型的字段而不是 `extra` 里的
  约定键，C1 的步骤替换内容时用 `dataclasses.replace` 保留它。为 `None`（手工构造）时，单个无 `extra` 的 `Text`
  输出成字符串，其它输出成列表。
- **共享而不复制**：adapter 不深拷贝。`Opaque.raw` 和 `extra` 里的嵌套对象（`cache_control` 等）在输入、中立消息和
  输出之间是同一个对象；这是有意的（大块的图片、PDF 不复制）。约定：调用方和 lite-context 都不改动它们，要改就先
  复制。lite-context 自己从不改动输入（每个 adapter 测试都检查输入没变）。
- system prompt 和 tool 定义不是 `Message`：作为 `fixed` 文本（或直接给 token 数）传给 `fit()`，计入预算，永不裁剪。
- `Opaque` 原样保留 adapter 不认识或不该动的内容（图片、PDF、thinking 块、厂商专有块），`raw` 是原始对象，
  转回去时原样输出。`tokens` 是它的成本；调用方知道时直接填上，否则由计数器按种类和 adapter 在转换时取出的中立度量估（§4）：
  `text`（thinking 块、文本文档自己的文字）、`pages`（PDF 页数，转换时数一次）。`tokens is None` 就是"估算的"；**永远不对 `raw` 跑文本计数**（base64 会被算成几十倍）。报告列出每个
  估算出来的 `Opaque`。
- **thinking 块不能改**：Anthropic 要求带 tool_use 的那一轮的签名 thinking 块原样回传。任何步骤都不 mask、不截断
  `kind="thinking"`；它们只按 lite-agent §6 的规则在最近一次 `Compaction` 之前整体去掉。
- **配对**：tool call 和它的结果用 `id` / `call_id` 配对。一次 assistant 消息里的多个 call 是一个**批次**，它们的结果
  可能分在连续几条消息里（OpenAI 每个结果一条 `role: tool`）。§5 的每一步都以整个批次为单位，不拆开。
- **轮**：一轮从一条带 `Text`、不带 `ToolResult` 的 user 消息开始（人说的话），到下一条这样的消息之前。只装 tool
  结果的 user 消息不开始新的一轮。`DropOldest`、`plan_compaction` 都用这个定义。
- **不完整的配对**：最后一个批次的结果还没到（loop 中途调用 `fit()`）是正常的，原样输出。除此之外，没有结果的
  call、没有 call 的结果、同一个 call 的第二个结果都是输入错误：`fit()` 抛 `BrokenPairing`，不猜、不补。一个批次
  还有 call 没回答时，下一条消息必须是带 tool 结果的消息（两家 API 都这样要求），所以批次不会被一句人话从中间
  隔开，按轮切也就不会拆开批次。

### 3.2 Adapter

一个 adapter 是一对函数：`to_messages(native) -> list[Message]`、`from_messages(list[Message]) -> native`。

| Adapter | 位置 | 说明 |
| --- | --- | --- |
| `anthropic` | 核心（只处理 dict，不 import SDK） | Messages API 的 `messages` 数组（system 在 Anthropic 里本来就是单独的参数）：string content、`text` / `tool_use`（`input` 必须是能原样往返的纯 JSON 对象：没有非字符串键、tuple、NaN / Infinity）/ `tool_result` 块（`tool_result` 的 content 可以是字符串或块列表，`is_error`）、`cache_control`；其它块（`image`、`document`、`thinking`、`redacted_thinking` …）变 `Opaque` |
| `openai` | 核心（只处理 dict） | Chat Completions 的完整 `messages` 列表：`system` / `developer` 消息拆出来作为 `fixed` 返回并在转回时放回原位；assistant 的 `tool_calls`（只接受 `type: "function"`；`arguments` 原样进 `raw_args`；`tool_calls` 为 `null` 或 `[]` 时原样保留，SDK 的 `model_dump()` 会给出 `null`）、`content: null` 或没有 content；每条 `role: tool` 消息变一条只含一个 `ToolResult` 的 user 消息，转回时还是各自一条 |
| `lite_model` | extra `lite-model` | lite-model 的 turn 类型；版本跟 workspace 的 lite-model pin |
| 宿主自有类型 | 宿主里写 | 约 30 行；lite-context 文档里给一个例子 |

规则：**往返无损**。对上表写明的输入范围，`from_messages(to_messages(x)) == x`，`Opaque` 块逐字节原样回来。
每个 adapter 都有这条性质测试，并且下面每种情况各有一个具名用例：字符串 content 与单个 text 块；`cache_control`
在块上和消息上；`tool_result` 的 content 是块列表（文字加图片）、`is_error`；格式错误的 `arguments` 字符串；
`content: null` 与 `""`；一个批次的多条 `role: tool`；`name` 字段；`system` / `developer` 的位置。表外的输入
（不认识的 role 等）抛 `UnsupportedInput`，不尽力转换。`key` / `pinned` 不在厂商格式里，由调用方在转换后设置
（`with_keys()` 辅助函数）；pin 一个 call 时它所在的整个批次和结果一起 pin。

OpenAI 的 tool 消息没有错误标记，`ToolResult.is_error` 转成 OpenAI 格式时不出现。OpenAI adapter 的
`Converted.fixed_texts()` 把 system / developer 消息变成 `fit(fixed=...)` 要的文本。一条中立 user 消息里混着
tool 结果和别的内容时，OpenAI 输出按原顺序分段：每个结果一条 tool 消息，每段其它内容一条 user 消息。

## 4. 计数与预算

```python
class Counter(Protocol):
    def text(self, s: str) -> int: ...
    def opaque(self, part: Opaque) -> int: ...       # used when part.tokens is None
    per_message: int                                 # framing overhead per message

CharEstimate(chars_per_token=3.5, opaque_default=1500, per_message=4)   # default, no dependency
TiktokenCounter(encoding="o200k_base", ...)                             # extra `tiktoken`

Budget(limit: int, reserve_output: int)   # limit = the model's input window as the caller decides it
```

- 默认计数器是估算（字符数 / 3.5），与宿主 A 现在的做法一致；宿主 B 可以换成 tiktoken。厂商 SDK 的 count_tokens
  是一次网络调用，不进核心（§15）。
- `tokens` 不为 `None` 时直接用，计数器不被调用（`count_message` 里判断，每个 `Counter` 不用各自记得）。否则
  `CharEstimate.opaque()` 只看 `kind` 和中立度量，从不读 `raw`：`image` 1600；有 `pages` 时页数 × 2000；有 `text`
  时数文字；其它（`redacted_thinking`、读不出页数的 PDF、未知块）用默认值 1500。几个数都是构造参数。
  adapter 取度量：Anthropic 的 `thinking` 文本、文本来源的 `document`、base64 PDF（数 `/Type /Page`）；OpenAI 的
  `file` 部分里 `data:application/pdf;base64,` 开头的 `file_data`。页数是数 `/Type /Page` 的估计：页对象放在压缩
  对象流里（PDF 1.5 的 `ObjStm`）时数不到、落到默认值；增量更新过的文件可能多数。都只影响估算，不影响内容。
- `CharEstimate` 的参数必须为正（`chars_per_token`）或不为负（其它）；`fixed` 里的 token 数必须是非负的 int
  （`bool` 不算），否则 `ValueError`。
- `TiktokenCounter` 推迟到 C4、宿主 B 接入时做：tiktoken 第一次用时从网络下载编码文件，和"测试不联网"冲突，需要
  先定下编码文件怎么随包提供；宿主 B 也可以直接实现 `Counter` 协议。
- 预算判断用估算，记账用厂商返回的 usage（宿主 A 的决策 D-10 也是这样）。报告里的数字标明是估算。
- 可用量 = `limit - reserve_output - fixed`。`fixed` 本身就超出时直接 `DoesNotFit`，不做任何裁剪。

## 5. 历史：`fit()`

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

### 5.1 流程

1. 有 `state.compaction` 时先应用它（§6.2）：被覆盖的消息换成一条摘要消息。
2. 按 `state` 里记下的切点重放上次的削减（§5.4）。
3. **常驻步骤**（`always=True`，如 `TruncateToolResults`）对每条消息都执行，不看是否超预算。它对同一条消息的结果
   永远一样，所以不会破坏 prompt cache 的前缀。
4. 仍超出高水位时，按顺序执行**削减步骤**，每一步都削到低水位（`target`，默认预算的 70%）为止，没超就跳过；
   新的切点写进返回的 `state`。
5. 全部执行完仍超预算 → `DoesNotFit`；报告里有每一步做了什么。
6. 高水位默认 = 预算的 90%。`report.compaction_due` 在第 1–4 步**之后**计算：应用了现有摘要和全部削减、仍超过
   高水位时为真，提示调用方该写新摘要了（§6）。

### 5.2 不变的规则（每条都有测试）

1. 输入的 `messages` 和 `state` 不被修改；同样的输入得到同样的输出（纯函数）。
2. 一个批次的 `ToolCall` 和它们的 `ToolResult` 要么都在，要么都不在；mask 只换结果的内容，不删配对。
   不完整的配对按 §3.1 处理。
3. `pinned` 的消息永不被 mask、drop 或摘要覆盖。
4. 最后一轮的人话（最后一条带 `Text`、不带 `ToolResult` 的 user 消息）永不被动。最新的 tool 结果**不受**这条
   保护：常驻步骤可以截断它，并写进报告；否则一个超大的结果会让每次调用都 `DoesNotFit`。
5. 报告列出每个被动过的消息（`key`，没有 key 时用位置）、动作和省下的 token；没被列出的消息与输入逐字节相同。
   这条在运行时检查：每个步骤执行后，`fit()` 比对它的输入和输出，有被改或被删、却不在这个步骤报告里的消息就抛
   `UnreportedChange`。宿主自己写的步骤同样受这个检查。
6. 每个占位文本是固定的、写在步骤里的常量，测试逐字比对。
7. `kind="thinking"` 的 `Opaque` 不被任何步骤改动（§3.1）。

### 5.3 步骤

| 步骤 | 类型 | 做什么 |
| --- | --- | --- |
| `TruncateToolResults(max_tokens, keep="head" \| "head_tail")` | 常驻 | 超长的 tool 结果截到上限，加 `[cut: N more characters not shown]`；加上标记反而更长时不截。上限按整个结果算：一个结果有几段文字时，超了就用换行把它们接成一段（位置和 `extra` 用第一段的）再截；`head_tail` 的尾部只从头部剩下的部分里取，两段不会重叠 |
| `MaskToolResults(keep_last=N)` | 削减 | 从最旧的**批次**开始，把批次里每个 tool 结果的内容换成 `[result cleared to save context; call {name} again if you need it]`，最近 N 个批次不动；结果里的图片一并换掉 |
| `MaskOpaque(keep_last=N)` | 削减 | 旧的 `image` / `document` 块（消息里的，和 tool 结果里的）换成 `[{kind} removed to save context]`（宿主 B 现在对附件的做法）；不碰 `thinking`。单位是"一条带图片或文档的消息"，一条消息里的都一起削；`keep_last` 数的是最近几条这样的消息 |
| `DropOldest(keep_last_turns=N)` | 削减 | 删掉最旧的整轮（§3.1 的定义，所以批次不会被拆开），在第一条被删消息的位置留一条 `[N earlier messages not shown]`（N 是删掉的消息数）。含受保护消息的轮不删。默认不在步骤里，调用方明确要才加 |

所有步骤都不碰受保护的消息（pinned 的、最后一轮人话那一条）。

**宿主自己的步骤**（C1a 实现的协议）：步骤在 `Entry(ref, message)` 上工作，`ref` 是调用方的 key，没有 key 时是
输入里的位置 `#i`，新插入的消息用自己的 ref（如 `+drop_oldest`）。

- 常驻步骤（`AlwaysStep`）：`name`、`always = True`、`apply(entries, ctx) -> (entries, actions)`。
- 削减步骤（`CuttingStep`）：`name`、`always = False`、`units(entries, ctx) -> [Unit]`（从旧到新、可以削的单位，
  `Unit.refs` 是它覆盖的消息；`Unit.at` 是步骤自己的定位）、`apply(entries, units, ctx)`（只削给它的这几个单位）。
  `fit()` 决定削几个：先重放 `state` 里切点及之前的单位，超过高水位时再往后加到低水位。加的时候只停在消息的边界上
  （同一条消息的几个单位一起削，因为切点记的是整条消息）；不假设"削得越多越省"（占位可能比原文长）：一次把所有
  单位都削掉、从被动过的消息算出每个单位省多少，用累加和估出每个个数的总数，取最少能到低水位的个数，到不了就取
  最省的那个；再真正削一次、精确地数，估少了就往后试。都不如重放省的话保持重放。
- `StepContext` 给出计数器和受保护的 ref。
- 每个步骤执行后 `fit()` 检查：受保护的消息和 thinking 块没变（`ProtectedChange`）；没报告的消息原样、原序都在，
  新增的消息都报告了（`UnreportedChange`）；配对仍然完整（`BrokenPairing`）。
- 每个候选输出都要检查，包括 `fit()` 二分查找时试过、最后没用上的，所以违反规则的步骤一定会被发现。
- `lite_context.testing.check_step(step)` 把宿主的步骤放进 `fit()`，在几段样例对话（并行批次、没完成的批次、
  pinned、图片、thinking）和几个预算上跑，检查上面这些规则、两次调用结果相同、输入没变；再把每段对话一轮一轮
  条地加长（三个预算）、每次传回 `state`，检查切点没动的那几次，上一次的请求是这一次的前缀（cache 不被破坏；刚离开
  保护的那条消息除外，§5.4）；再跑一次没有 key 的历史。返回步骤实际改动过的次数。样例里有单独的图片、一条消息里
  两张图片、并行批次里每个结果各带一张图片。

### 5.4 状态：为什么削减要记下来

`fit()` 是纯函数，只看输入。如果不记下上次削到了哪里，原始历史一旦超过高水位，之后每一轮都会重新削到低水位，
而历史每轮都在变长，于是每轮都多 mask 一条旧消息，prompt cache 每轮都从那里失效。所以每个削减步骤的结果是一个
**切点**：它削到了输入里的哪一条消息（`through`），加上输入从开头到这条消息的 ref 的摘要值。重放时，这个步骤
在 `through` 及之前的所有单位都削掉。切点按位置而不按"前几个单位"记：前面的步骤删掉了某些单位，或者某条消息
离开了保护（最后一轮人话换了一条），单位列表会变，但位置不变，所以不会误报 `StaleState`（C1a review 发现）。
代价：离开保护的那张旧图片会在下一次重放时被削掉，旧消息因此改动一次。单位的先后由 `fit()` 按输入位置排，
不依赖步骤列出的顺序。切点存在 `ContextState` 里：

```python
@dataclass(frozen=True)
class ContextState:
    cuts: Mapping[str, Cut]       # step name -> Cut(through: ref, digest of the input refs up to it)
    compaction: Compaction | None # C1b
```

- 下一次调用传回 `state`：已有的切点原样重放；只有在重放后仍超过高水位时，切点才往前推。切点只往前，不后退。
- 往前推的切点如果让总数反而变大（`DropOldest` 的占位比删掉的几条短消息还长），就不采用，保持重放的结果。
- 调用方把 `state` 存在历史旁边（`to_json()` / `from_json()`，一列 JSON 即可；格式带版本号，不认识的版本报错）。
  不传 `state` 等于从零开始，结果仍然正确，只是 cache 不稳定。
- 切点和 `Compaction` 一样带被覆盖消息 key 的摘要值；历史在切点之前被改过时抛 `StaleState`（§6.2）。
- **切点量的是应用摘要之后的历史**。一次新的压缩（§6.1 `accept`）返回的 `state` 不带切点：被摘要覆盖的单位已经
  不在了，旧切点的摘要值对不上；而且压缩本身已经让 prompt cache 从开头失效，从零开始削不损失什么（C1a 收尾时定，
  见阶段记录）。
- key 不能以 `#` 或 `+` 开头（那是 lite-context 内部的 ref），否则 `ValueError`。
- 切点要靠 key：消息没有 key 时 `fit()` 照样能削，但返回的 `state` 不记切点；带着切点的 `state` 配上没有 key 的
  消息抛 `StaleState`。这一次调用的步骤里没有的切点不带进新 `state`。

## 6. 摘要：`Compaction`

```python
@dataclass(frozen=True)
class Compaction:
    covers_through: str      # key of the last message the summary covers
    covered_digest: str      # sha256 over the covered messages' keys, in order (after the pinned prefix)
    text: str                # exactly what the model is shown, preamble included
    provenance: Mapping[str, object]   # role / vendor / model / usage, as the caller's LLM layer reports them
    created_at: str          # ISO 8601
```

`Compaction` 存在 `ContextState.compaction` 里，和切点一起序列化（`to_json()` / `from_json()`）。

### 6.1 写摘要

```python
if fitted.report.compaction_due:
    plan = plan_compaction(messages, state=saved_state, policy=SummaryPolicy(keep_last_turns=4))
    if plan is not None:                     # None: nothing new can be covered
        reply = my_llm.call(role="compact", system=plan.system, messages=plan.prompt)  # the caller's role
        new_state = plan.accept(reply.text, truncated=reply.stopped_at_limit, provenance=reply.provenance)
        store(new_state)                     # raises EmptySummary / TruncatedSummary instead
```

- `plan.covered` 是摘要将代表的消息的 key（从 pinned 前缀之后算），`plan.covers_through` 是最后一个；覆盖范围的
  终点永远是某一轮的开头（`keep_last_turns` 至少为 1，当前这一轮和没完成的批次永远不进摘要；轮数不比要保留的多、
  或者没有人话的历史，都返回 None）。
- `accept` 的 `provenance` 按 JSON 往返之后的样子保存（tuple 变 list），不是 JSON 的值立刻 `ValueError`，不等到
  存储时才失败。
- 便捷函数 `await acompact(messages, state=..., summarize=fn, policy=...)` 把三步合在一起；`fn(system, prompt)` 是
  调用方给的 async 函数，返回 `Summary(text, truncated, provenance)`；没有新内容可覆盖时返回 None。
- `accept` 返回的 `state` 只有新摘要、没有切点（§5.4）。
- **覆盖范围**：从历史开头的 pinned 前缀（比如最初的任务说明）之后开始，到切点为止。范围里不能有 pinned 消息：
  切点最远到下一条 pinned 消息之前，所以 pinned 消息永远留在原来的位置，顺序不变。
- 切点只落在轮的边界上（§3.1），所以不会拆开批次。后一次压缩从头覆盖，把前一次的摘要也吃进去；被截断或为空的
  摘要算失败，不能顶替历史。这几条沿用 lite-agent S3b。
- **`SummaryPolicy` 是可替换的**：`keep_last_turns`、`render(previous, messages) -> (system, transcript)`、
  `wrap(mark, reply) -> text`、`unwrap(text)`。`mark` 每个 plan 定一次、记在 plan 上，`wrap` 只去掉带这个 mark 的
  头标行（任何字符串的 mark 都行）；前一次的摘要去掉前言再交给模型，免得前言被抄进新摘要、再套一层。默认 policy 照搬 lite-agent 实测过的做法（它的 §15 S3b）：transcript 的每一段以带随机
  token 的头标开头（tool 结果里的文字冒充不了用户）；说明在 transcript 之后再说一遍（小模型否则会回答 transcript
  里最后一个问题，6 次里 6 次）；去掉回复里抄进来的头标行；thinking 块不进 transcript，图片写成 `[image]`；前一次的
  摘要作为第一段 `earlier summary` 放进去。`mark` 可以传入（测试用），默认每次随机。lite-agent 在 C7 提供它自己的
  policy，从而保持它现有的行为。
- 需要 `key`：没有 key 的历史不能压缩（报错）。key 只要求在一份历史里唯一，**不要求可比较**（UUID 也行）。

### 6.2 应用摘要

`fit(..., state=s)`，`s.compaction = c`。应用摘要在 `fit()` 里是一个内部步骤：它像别的步骤一样报告动作（被覆盖的
消息 `summarised`、摘要消息 `inserted`），经过同一个运行时检查，不另写一份规则：

1. 找到 `c.covers_through` 的位置；找不到 → `StaleState`。
2. 从 pinned 前缀之后到这个位置的消息，它们 key 的摘要值必须等于 `c.covered_digest`（条数变了摘要值也会变，所以
   不另存条数，C1b 收尾时去掉了 `covered_count`）；
   不相等（切点之前有消息被插入、删除或改了 key）→ `StaleState`。不悄悄套用一个过期的摘要。
3. 这个位置必须是轮的边界，范围里不能有 pinned 消息；否则 → `StaleState`。
4. 输出：pinned 前缀原样，然后一条 user 消息（内容就是 `c.text`，ref `+compaction`），然后切点之后的消息。只用
   最近的一条 `Compaction`。之后的削减步骤在这个结果上工作，所以切点量的是摘要之后的历史（§5.4）。
5. 消息没有 key 而 `state` 里有摘要 → `StaleState`。

摘要消息是一条 user 消息，后面通常紧跟着下一轮人话（也是 user），前面有 pinned 前缀时还会更多：连续的 user 消息。
Anthropic 会把它们合并，OpenAI 接受；要求严格交替的后端（有些本地模型的 chat template）需要宿主自己合并。

切点之前的消息**内容**被改过（key 不变）时检查不出来；这是调用方的约定：被摘要覆盖的消息不再修改。要改就丢掉
`state`，重新压缩。

## 7. 长文档与检索结果：`lite_context.docs`

两个首批宿主各写了一遍的东西。模型调用都由调用方传入。规则来自 KB `global/map-reduce-llm-guide.md` §3 和宿主的
实测：先数，装得下就原样一次调用（切分路径的 prompt 和调用与不切时完全一样）；reduce 按预算把相邻笔记分组、一层层
合并，单条笔记合不了、合并结果超预算都让整个 reduce 失败，绝不退回到直接拼接；只有调用方判定为"太大被拒"的失败才
对半拆，别的失败原样抛出。

| 函数 | 做什么 |
| --- | --- |
| `split(text, max_tokens=, counter=, overlap=0.06, prefer="\n", heading=None) -> [Piece]` | 装得下就是原样一段。否则在每个 `prefer` 之后断开成小段（超限的小段先截到限内），每个小段只数一次；一段取尽可能多的整个小段，再用一次真实计数核对（不可加的计数器超了就从尾部退）；下一段往回重叠约 `overlap`，也从小段边界开始。小段计数相加对子词分词器是往高估，是安全的方向（KB 指南 §3）；总计数量约为全文的 3 倍。`heading` 是一个正则：每段带上它开始所在章节的标题行（开在标题行上就是这个标题；宿主 B 实测：不带时，切开后 60 条里 60 条丢了所属章节） |
| `pack(units, budget=, cost=, overhead=0) -> [[unit]]` | 把连续的单元按预算分组，不打乱顺序；单个单元自己就超预算时 `DoesNotFit` |
| `await halving(items, call, too_large=, min_size=1) -> Halved(calls)` | 调 `call(items)`；`too_large(exc)` 为真时对半拆开、先后调用；别的失败、到 `min_size` 还被拒、取消（`BaseException`，不交给 `too_large`）都原样抛出。`calls` 按顺序记下每次成功调用读的范围和它的结果，`results` 是拼起来的结果。`min_size` 至少为 1。文本按行拆：传 `text.splitlines(keepends=True)` |
| `await reduce_tree(notes, budget=, cost=, merge=, overhead=0, max_depth=8) -> Reduced(notes, levels)` | 一层层合并相邻笔记，直到能一起放进预算；返回的笔记由调用方用自己的最终 prompt 读（中间合并和最终 reduce 的 prompt 不同）。每层的组按顺序一个一个合并（限额账户上并发只会被拒）。没有进展、合并结果超预算、到 `max_depth` 都 `DoesNotFit` |
| `select(chunks, budget=, cost=, key=) -> Selected(chosen, duplicates, over_budget)` | 在调用方已经排好序的检索结果里按顺序取、按 key 去重；第一条放不下就停，之后不重复的都进 `over_budget`、重复的进 `duplicates`（排名低的短块不会挤掉排名高的；三个列表恰好分完输入） |

单元、笔记、检索块都是泛型：调用方用自己的类型（页、`(范围, 文本)`、检索结果），给出 `cost`。

## 8. `Transcript`：给没有存储的宿主

```python
t = Transcript(store=InMemoryStore(), id="thread-1")      # or PgStore (extra `pg`)
seq = t.append([user("...")], expected_seq=t.last_seq)
fitted = t.project(budget=..., steps=[...])              # latest ContextState + fit()
reply = my_llm.call(fitted.messages)
t.append([assistant_from(reply), fitted.state], expected_seq=seq)   # the state is an entry too
await t.compact(summarize=fn)                            # writes a new ContextState entry when due
```

- append-only：条目是 `Message` 或 `ContextState`，带单调递增的 `seq`（就是 `key`）；`append(entries, expected_seq)`
  并发冲突时一个成功一个被拒（`SeqConflict`），一批条目全有或全无。
- `project()` 是纯投影：同一份条目得到同一份输出。
- PostgreSQL：数据库和 role `lite_context`，schema `lite_context`；测试只用同一数据库里的 schema
  `lite_context_test`，URL 来自 `LITECONTEXT_TEST_DATABASE_URL`。表上有拒绝 UPDATE / DELETE 的 trigger（与
  lite-agent 的 events 表一样）。

## 9. REST 壳

`litecontext serve`（extra `server`，FastAPI），只绑 127.0.0.1:8750，拒绝别的地址；不做鉴权，与 lite-agent 的
HTTP 壳一样，等有宿主要从别的机器访问时再定。壳里没有逻辑：解析请求、调库、序列化。

| 端点 | 状态 | 对应 |
| --- | --- | --- |
| `POST /v1/fit` | 无状态 | `fit()`；`format: anthropic \| openai \| neutral` 决定 messages 的格式 |
| `POST /v1/compaction/plan` / `POST /v1/compaction/accept` | 无状态 | §6.1 的前后两步：返回写摘要的 prompt；交回摘要文本，返回 `Compaction` |
| `POST /v1/docs/split`、`/v1/docs/pack`、`/v1/docs/select` | 无状态 | §7 里不调模型的三个 |
| `POST /v1/transcripts`、`POST /v1/transcripts/{id}/entries`、`POST /v1/transcripts/{id}/project` | 有状态（需要 `pg`） | §8 |
| `POST /v1/transcripts/{id}/compaction/plan` / `.../compaction/accept` | 有状态 | §8 的 `compact()` 拆成两步：服务返回 prompt，客户端写好摘要交回，服务追加新的 `ContextState` 条目 |

`halving` / `reduce_tree` 需要回调，不在 REST 里。服务端自己调 role 写摘要（需要 lite-model 和凭证）是以后的选项（§15）。

## 10. 配置

库的参数都在代码里传。REST 壳的配置来自环境变量，前缀 `LITECONTEXT_`（`DATABASE_URL`、`HOST`、`PORT`）；
`.env.example` 是模板，`.env` 不进 git。

## 11. 测试

- pytest，`tests/conftest.py` 拒绝一切 socket（PostgreSQL 测试除外，只连 `LITECONTEXT_TEST_DATABASE_URL`）。
- §5.2 的每条规则至少一个有名字的测试；adapter 有往返性质测试（hypothesis）。
- 断言具体内容：输出的每条消息、报告里的每个动作和数字，不只是"没报错"。
- CI：ruff、ruff format、mypy strict、pytest，在 Python 3.11 和 3.12 上各跑一遍。
- 效果测量（C4 起）：宿主接入前后，记录同一批真实调用的输入 token、超窗口失败次数、prompt cache 命中；
  masking 和摘要的质量对比用 lite-eval 跑（§15）。

## 12. 与其它项目的关系

| 项目 | 关系 |
| --- | --- |
| 宿主 A（Python 3.12，uv，自有 LLM 层和类型，Temporal 里的 tool loop） | C4：tool loop 每一步先 `fit()`（`TruncateToolResults` + `MaskToolResults`），代替现在只在产生时截断；纯函数，留在 workflow 里；摘要走一个新的 activity。文档的按页分组和对半重试换成 §7 |
| 宿主 B（Python 3.11，pip，Anthropic 格式的 dict，chat 历史在 PostgreSQL 里由 TypeScript 服务持有） | C4：chat 请求进 Python 时先 `fit()`；`ContextState` 存在 thread 上（宿主自己的 migration）；文档的切分、对半重试、分层 reduce 换成 §7 |
| lite-agent | C7：依赖 lite-context，把事件映射成中立 `Message`，它的 `SummaryCompactor` 和投影里的遮盖规则改用 lite-context。需要改它的设计（规则 8：lite-model 是唯一的 lite 依赖） |
| lite-model | 不依赖。`lite_model` adapter 是 extra，版本跟 workspace pin |
| lite-rag、lite-mem | 不依赖。lite-rag 负责检索和排序，§7 的 `select` 只在它排好的结果里挑 |
| lite-eval | C4 之后用它对比 masking 和摘要的质量（extra `eval`） |

安装方式：GitHub Release wheel（与 lite-model、lite-agent 一样）。uv 宿主在 `[tool.uv.sources]` 写 wheel URL；
pip 宿主在 requirements 里写 `lite-context @ <wheel URL>`。

## 13. 项目结构

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

## 14. 阶段与验收

§5.2、§6 的每条规则都要对应到至少一个有名字的测试；阶段收尾时列出"规则 → 测试"的对照。

每个阶段收尾都按 AGENTS.md Workflow 第 4 步执行：`/simplify`，然后 Codex 只读 review，然后修复（否决的
写在本节的阶段记录里），最后重跑全部检查。

| 阶段 | 内容 | 验收 |
| --- | --- | --- |
| S0 | 仓库、设计文档、AGENTS.md / CLAUDE.md / README、项目 KB、workspace `AGENTS.md` 登记、端口 8750 登记 | 本文经 owner 确认；Codex 对设计的 review 已处理 |
| C0 | 骨架：pyproject（uv、uv_build、ruff、mypy strict、pytest、`requires-python >=3.11`）、CI（3.11 + 3.12）、拒绝 socket 的 conftest；§3 的类型和 `anthropic` / `openai` adapter；§4 的计数器和预算；`Report`、`DoesNotFit` | 两个版本上 ruff、mypy、pytest 全绿；adapter 往返性质测试加 §3.2 列出的每个具名用例；表外输入抛 `UnsupportedInput`；`Opaque` 的估算按种类且不对 `raw` 计数；核心 import 时没有第三方模块（测试按 §2 的清单检查 `sys.modules`）；`fixed` 超预算直接 `DoesNotFit` |
| C1（分两步：C1a = `fit()`、步骤、`ContextState` 与切点、运行时检查、`testing.check_step`；C1b = §6 的摘要） | `fit()` 与 §5.3 的四个步骤、高低水位、`ContextState` 与切点（§5.4）、§6 的 `plan_compaction` / `accept` / `acompact`、`SummaryPolicy` 与应用摘要 | §5.2 七条规则各有测试（含 `UnreportedChange` 抓住一个故意不报告的自写步骤）；同一输入两次 `fit()` 结果相同；带 `state` 连续调用 20 轮、历史每轮变长：被改动的旧消息只在越过高水位的那几轮变化，其余轮次与上一轮的输出前缀逐字节相同（cache 稳定性）；不完整配对的三种情况（最后批次未完成放行、缺结果和缺 call 抛 `BrokenPairing`）；`keep_last` 按批次计数；`DropOldest` 不拆开以 tool 结果开头的轮；超大的最新 tool 结果被常驻步骤截断而不是 `DoesNotFit`；thinking 块不被任何步骤改动；`StaleState` 的三种情况（找不到切点、数量或摘要值不符、切点不在轮边界）；摘要后原 messages 不变、输出以摘要开头；后一次压缩吃进前一次；截断或空摘要失败；切点不拆配对、不越过 pinned 消息、pinned 前缀留在摘要之前；找不到 `covers_through` 报错；`testing.py` 的 fixture 能套在一个宿主自写的步骤上 |
| C2 | §7：`split`、`pack`、`halving`、`reduce_tree`、`select` | 用两个首批宿主现有实现的行为写测试（重叠比例、切在换行、单元超预算报错、对半到 `min_size` 后抛出、分层次数、去重）；`reduce_tree` 在 `merge` 不缩短输入、单条超预算、到 `max_depth` 时都抛出而不是死循环；调用方传入的 fake 记录每次调用的输入，断言分组和顺序 |
| C3 | 发布 v0.1.0 release wheel | 一个新的 3.11 pip 环境和一个 3.12 uv 项目都能从 wheel 安装并跑通 README 的例子 |
| C4 | 宿主 A、B 同时接入（各自仓库里做，各自收尾）；lite-context 这边只修接入中发现的问题 | 两个宿主的全部测试通过；接入前后同一批调用的输入 token、超窗口失败、cache 命中写进 KB；宿主 B 的 `ContextState` 持久化有迁移和测试；**通过条件**：用接入前录下的一批真实请求重放，超窗口失败为 0，输入 token 不高于接入前，两个宿主各自的测试全绿 |
| C5 | §8：`Transcript`、`InMemoryStore`、`PgStore`（extra `pg`）、`litecontext migrate` | append-only（trigger 拒绝 UPDATE / DELETE）；并发 append 一个成功一个被拒；内存和 PG 的投影相同；跨进程读回后投影相同 |
| C6 | §9 REST 壳（extra `server`） | 每个端点与库函数的输出逐字段相同（同一组 fixture 跑两遍）；只绑 127.0.0.1；请求体超限被拒 |
| C7 | lite-agent 改用 lite-context（在 lite-agent 仓库里，先 plan mode）：它提供自己的 `SummaryPolicy`（轮的计法、渲染、头标处理） | lite-agent 的全部压缩测试不改一行仍然通过；做不到的地方先在 lite-context 补能力，或在 lite-agent 的阶段记录里逐条写明为什么改测试；它的设计 §6 和规则 8 同步更新 |
| 按需 | token 级压缩步骤、服务端摘要、tiktoken 以外的计数器、async 计数器 | — |

### 阶段记录

（每个阶段收尾时追加：做了什么、规则 → 测试对照、Codex 的发现与处理。）

- **S0（2026-11-26）**：仓库、本文初稿、AGENTS.md / CLAUDE.md / README、项目 KB、workspace 登记、端口 8750。
  Codex 不可用（账户额度用完，到 2026-12-18 才恢复），改由一个不带本文作者结论的 Claude 子代理做对抗式设计 review，
  12 条发现全部成立、全部采纳：
  1. 纯函数 `fit()` 加高低水位不能让 cache 稳定 → 加 `ContextState` 和切点（§5.4）。
  2. "轮"按 user 消息切会拆开以 tool 结果开头的批次 → 重新定义轮（§3.1）。
  3. "最后一条 user 消息不动"与常驻截断冲突 → 改为只保护最后一轮的人话（§5.2-4）。
  4. 中立类型装不下厂商格式的全部字段 → `raw_args`、结果内容可含图片、`extra`、写明往返范围和具名用例（§3）。
  5. mask thinking 块会被 Anthropic 拒绝 → thinking 不被任何步骤改动（§3.1、§5.2-7）。
  6. `Opaque` 的估算 → 按种类估、不对 `raw` 计数、报告标明（§3.1）。
  7. key 不可比较、历史被改后摘要过期、pinned 在覆盖范围里 → `covered_count` + 摘要值 + `StaleState`、覆盖范围不含 pinned（§6）。
  8. 不完整配对没定义 → §3.1 写明，`BrokenPairing`；`keep_last` 按批次。
  9. 自写步骤能悄悄丢内容 → 运行时 `UnreportedChange` 检查（§5.2-5）。
  10. `reduce_tree` 可能不收敛 → 终止条件（§7）。
  11. C7 与 lite-agent 的摘要规则不一致、C4 没有通过条件 → `SummaryPolicy` 可替换、C4 写明通过条件（§6.1、§14）。
  12. 零碎矛盾（`expected_seq`、REST 缺压缩端点、测试 schema、核心模块清单、`compaction_due` 的计算时机）→ 逐条改正。
- **C0（2026-11-26）**：`pyproject`（uv_build，与其它 lite 模块一致，原文写的 hatchling 改掉；`requires-python
  >=3.11,<3.13`，ruff `target-version = py311`，mypy `python_version = 3.11`）、CI（3.11 + 3.12 矩阵）、
  `types` / `count` / `report` / `errors`、`adapters/anthropic`、`adapters/openai`。与本文的偏差：`Opaque` 的估算放在
  计数器里而不是 adapter（§3.1、§4 已改；收尾时又改回由 adapter 取度量，见下面的 /simplify）；`TiktokenCounter`
  推迟到 C4（§4）。
  规则 → 测试：
  - 往返无损：`test_adapter_anthropic.py::test_every_supported_conversation_round_trips`、
    `test_adapter_openai.py::test_every_supported_conversation_round_trips`（hypothesis），加 §3.2 每个具名用例
    （字符串与单个 text 块、`cache_control`、tool_result 的三种 content 和 `is_error`、格式错误的 `arguments`、
    `content: null` / `""` / 没有键、并行批次的多条 tool 消息、`name` 和未知键、system / developer 的位置）。
  - 表外输入被拒：两个 adapter 的 `test_input_outside_the_supported_range_is_refused`。
  - `Opaque` 原样回来：`test_blocks_the_core_does_not_handle_become_opaque_and_come_back_unchanged`（同一个对象）。
  - 按种类估、不对 `raw` 计数：`test_count.py` 的 image / thinking / 文本文档 / PDF / 未知块五条。
  - 核心零依赖：`test_core_imports.py`（子进程里按 §2 清单 import，检查 `sys.modules`）。
  - `fixed` 超预算：`test_fixed_content_over_the_budget_does_not_fit_before_any_cutting`。
  - 测试不联网：`test_no_network.py`。
  见红：新测试先在空包上全部失败；之后逐个破坏被测代码确认关键测试变红（核心多 import 一个第三方模块、关掉拒绝
  socket 的 fixture、Anthropic 不记录列表形式、PDF 页数恒为 0、OpenAI 提前解析 `arguments`），再恢复。
  `/simplify`（四个角度的子代理）：采纳——计数器不再读厂商的块结构，adapter 在转换时取出 `Opaque.text` /
  `Opaque.pages`（PDF 只解码一次）；content 形式从 `extra` 里带前缀的约定键改成有类型的 `content_form`，
  `is_error` 改成 `bool | None`，`tool_calls: []` 和 `function` 的多余字段原样留在 `extra`，去掉命名空间键；两个
  adapter 的重复（纯文本判断、text 块、opaque、字符串或列表）收进 `adapters/_common.py`；加 `fixed_texts()`；
  `Opaque.estimated` 删掉（`tokens is None` 即估算）；OpenAI 输出混合内容时保持顺序；`Report` 里 C1 才用的字段
  和 `Action` 删掉，C1 再加。跳过——计数结果缓存（C1 的 `fit()` 要按条累加而不是每删一条重数一遍，记在这里给 C1）；
  `tool_use` 的 input 每次 json 往返（有大 input 的宿主时再说）。新加的测试（度量、顺序、`tool_calls: []`、
  `fixed_texts`、`tokens` 不问计数器）都用破坏被测代码的方式见过红。
  代码 review：Codex 不可用，由一个子代理对抗式 review 代替。6 条发现全部成立、全部修复，回归测试先红后绿：
  1. Anthropic `tool_use.input` 有非字符串键、tuple、NaN 时被接受却回不来 → `allow_nan=False` 并核对 `loads(dumps(x)) == x`，
     不等就 `UnsupportedInput`（`test_tool_use_input_that_would_not_come_back_equal_is_refused`）。
  2. OpenAI 没有 `type` 的 tool call 输出时被补上 `type` → 只接受 `type: "function"`
     （`test_a_tool_call_that_is_not_a_function_call_is_refused`）。
  3. 输出和输入共享嵌套对象 → 写进 §3.1 的约定（有意不深拷贝）。
  4. SDK 的 `tool_calls: null` 被拒 → 接受并原样保留（`test_the_sdk_shape_with_tool_calls_none_survives`）。
  5. 性质测试的生成器漏了几个具名情况 → 补上（浮点数、message 和 tool_use 上的额外字段、document / redacted /
     未知块、`tool_calls` 的四种形式、`function` 的多余字段、`name`、tool 内容里的图片和文件、`refusal`）；
     额外用 3000 个例子各跑了一遍，没有失败。
  6. `CharEstimate`、`Budget` 接受不可能的数 → 构造时和 `available()` 里校验（两条参数化测试）。
  只记录不改：`restore` 在手工构造的 `extra` 与核心字段同名时让核心字段赢（adapter 自己产生的 `extra` 不会同名）；
  `Message` 因为 `extra` 是 dict 而不能 hash（C1 若需要再说）；PDF 页数估计的局限（§4）。

- **C1a（2026-11-26）**：`structure.py`（批次、轮、配对检查）、`steps.py`（`Entry` / `Unit` / `StepContext`、
  `AlwaysStep` / `CuttingStep` 协议、四个步骤、固定占位文本）、`state.py`（`ContextState`、`Cut`、JSON 带版本）、
  `fit.py`（流程、高低水位、重放与推进切点、每步之后的运行时检查）、`testing.py`（`check_step`）；新错误
  `BrokenPairing`、`StaleState`、`UnreportedChange`、`ProtectedChange`；`Report` 加上动作和前后数字。与本文的
  偏差：批次规则收紧（还有 call 没回答时，下一条必须是带结果的消息）；`DropOldest` 的占位放在第一条被删消息的位置；
  所有步骤都不碰受保护消息。
  规则 → 测试（`tests/test_fit.py`，除另注）：
  - §5.2-1 纯函数、输入不变：`test_the_input_is_untouched_and_the_same_input_gives_the_same_output`。
  - §5.2-2 批次不拆开：`test_mask_keeps_the_last_n_batches_counting_a_parallel_batch_once`、
    `test_an_unfinished_batch_is_never_masked`；配对检查 `test_structure.py::test_broken_pairing_is_refused`（五种）；
    步骤拆开配对被抓 `test_a_step_that_breaks_a_rule_is_caught_at_run_time[break-pair]`。
  - §5.2-3 pinned：`test_a_pinned_batch_is_skipped_and_the_next_one_masked`、
    `test_drop_oldest_never_drops_a_turn_holding_a_pinned_message`、`...caught_at_run_time[touch-pinned]`。
  - §5.2-4 最后一轮人话受保护、最新 tool 结果可截：`test_mask_opaque_removes_old_images_but_not_the_last_human_turns_or_thinking`、
    `test_the_newest_tool_result_is_truncated_rather_than_failing_every_call`。
  - §5.2-5 报告与运行时检查：`test_cutting_steps_stop_at_the_low_water_mark`（动作逐条比对）、
    `...caught_at_run_time[silent-edit]`、`test_testing.py::test_a_host_step_that_hides_a_change_fails`。
  - §5.2-6 固定占位：截断、mask、opaque、drop 四条测试都逐字比对常量。
  - §5.2-7 thinking：`test_a_step_that_removes_a_thinking_block_is_caught` 和 opaque 那条。
  - §5.1 高低水位：`test_nothing_is_cut_below_the_high_water_mark`、`test_cutting_steps_stop_at_the_low_water_mark`、
    `test_when_every_step_has_run_and_it_still_does_not_fit_it_raises_with_the_report`、
    `test_compaction_is_due_when_the_cut_history_is_still_over_the_high_water_mark`。
  - §5.4 状态：`test_with_state_old_messages_change_only_on_the_turns_that_cross_the_high_water_mark`（20 轮，cache
    稳定性）、`test_state_is_replayed_even_when_the_history_would_fit_without_it`、
    `test_state_from_a_history_that_changed_before_the_cut_is_stale`、`test_without_keys_no_cuts_are_kept`、
    `test_state_survives_json_and_an_unknown_version_is_refused`。
  - `check_step`：`test_testing.py`（四个内置步骤通过且确实改动过，隐瞒改动的宿主步骤失败）。
  见红：测试先写，在没有实现时全部失败；之后对 14 处关键代码逐一破坏（高水位门槛、低水位、重放、保护检查、未报告
  检查、thinking 检查、配对检查、摘要值、未完成批次、opaque 保护、drop 保护、keep_last、未回答的 call、二分查找），
  每处都有测试变红，再恢复。
  `/simplify`（两个子代理）：采纳——`fit()` 的削减循环原来对每加一个单位都重放并重数整段历史（1000 条消息首次
  调用 356–526 ms），改成每条消息只数一次、之后只重数步骤报告过的消息，并用二分查找定单位数（同样的输入 6–21 ms，
  动作数不变）；`MaskToolResults` 直接从批次的 call 消息里取工具名，不再每次重建批次；三个步骤共用 `_rewrite`；
  重复 key 检查共用 `check_unique`；测试的 `COUNTER` 用 `lite_context.testing` 里的；去掉死分支和一个 type ignore；
  thinking 检查只看被动过的消息。设计上定下：新压缩不带切点、切点量摘要之后的历史（§5.4）；应用摘要是走同一个
  运行时检查的内部步骤（§6.2），C1b 照此实现。跳过——把 `Unit` 做成泛型载荷（工具名改从消息里取之后没有需要）。
  代码 review（Codex 不可用，子代理代替）：9 条发现，8 条修复、1 条记录；回归测试先红后绿：
  1. 一个削减步骤推进后，后面步骤的单位列表变短，"前 n 个单位 + 摘要值"对不上，正常增长也抛 `StaleState` →
     切点改成"削到哪一条消息 + 输入到那里的 ref 摘要值"，按位置重放（§5.4）
     （`test_two_cutting_steps_keep_working_as_the_history_grows`，要求出现"一个步骤越过另一个步骤的切点"）。
  2. 保护离开一张旧图片后它插进单位列表中间，同样误报 → 同一个修复
     （`test_protection_moving_off_an_old_image_does_not_make_the_state_stale`）。
  3. 调用方的 key `#1` 和没有 key 的消息的位置 ref 撞上，计数算错，超预算却不报错 → 保留 `#`、`+` 前缀；检查时
     也要求 ref 唯一（`test_keys_that_look_like_internal_refs_are_refused`、`...caught_at_run_time[duplicate]`）。
  4. `DropOldest` 的占位比删掉的消息还长，本来装得下却 `DoesNotFit` → 不采用让总数变大的切点
     （`test_a_cut_that_would_cost_more_than_it_saves_is_not_taken`）。
  5. `check_step` 不考 state 和增长 → 加增长与无 key 的运行；`fit()` 自己按位置排单位，所以"新到旧列出单位"
     不再是错误（`test_the_order_a_step_lists_its_units_in_does_not_matter`）；增长检查抓得住输出随历史漂移的步骤
     （`test_a_step_whose_output_drifts_as_the_history_grows_fails`）。
  6. 截断按段而不按整个结果 → 按整个结果（`test_truncation_limits_a_whole_result_not_each_text_part`）。
  7. `head_tail` 在不可加的计数器下头尾重叠 → 尾部只从剩下的部分取
     （`test_head_and_tail_never_overlap_even_with_a_counter_that_is_not_additive`）。
  8. 只记录：`DropOldest` 的占位是一条带文字的 user 消息，会被当成一轮的开头；`fit()` 里只有 `DropOldest` 自己按轮
     切、而且看的是它的输入，所以现在没有影响。
  9. `from_json` 对畸形输入抛各种异常 → 统一 `ValueError`（`test_a_malformed_state_is_a_value_error`）。
  另外：二分查找时试过的每个候选都要过运行时检查（否则一个被放弃的违规输出不会被发现）。
  测试 1 的最终版本（预算 1200、15 轮）没有在旧代码上跑过；它的第一版（预算 600）和测试 2 在旧代码上是红的。
  修复后的计时与修复前相同量级（1000 条消息、三个削减步骤首次 27 ms）。
  第二轮 review（切点改动较大，按收尾规则再审一次；子代理写了随机增长探针：随机历史一条一条加长、随机步骤组合和
  预算、每次传回 state）：6 条发现全部修复，回归测试先红后绿：
  1. 切点记整条消息，但 `MaskOpaque` 一张图片一个单位，可能只削了一条消息里的一部分，下次重放把其余的也削了 →
     `fit()` 只在消息边界上停（`test_a_host_steps_units_in_one_message_are_cut_together`，宿主自己的按图片步骤）。
  2. `MaskOpaque` 的 `keep_last` 边界落在一条消息中间，同样的问题 → 单位改成一条消息
     （`test_images_in_one_message_are_cut_together_so_replay_cuts_no_more`）。
  3. 二分查找假设"削得越多越省"，mask 比占位还短的结果会变长，找不到其实存在的切法 → 改成上面的估算加验证
     （`test_a_cut_is_found_even_when_masking_some_results_makes_them_longer`、
     `test_when_no_cut_reaches_the_low_mark_the_cheapest_one_that_fits_is_taken`）。线性扫描时首次调用退回到
     392–667 ms，改成估算加验证后 10–15 ms（1000 条消息）。
  4. 同一个 `Text` 对象出现两次时截断结果重复 → 按位置找（`test_truncation_handles_the_same_text_object_twice`）。
  5. 截断加标记后反而更长 → 不截（`test_truncation_never_makes_a_result_longer`）；原来两条截断测试的输入因此改长。
  6. `check_step` 的样例里没有一条消息多张图片 → 加上，并且一条一条地加长、三个预算。
  7. 只记录：离开保护的图片在下一次重放时被削（§5.4 接受）；`check_step` 和随机增长测试对这一条放行。
  随机增长探针收进仓库：`tests/test_growth.py`（150 个种子，每次检查无异常、同输入同输出、切点没动时前缀不变）；
  另外手动多跑了 1000 个种子，没有失败。
  见红补充：消息边界、最省回退、单位排序、唯一 ref、不采用更贵的切点这几处都用破坏代码的方式确认有测试变红。

- **C1b（2026-11-26）**：`compaction.py`（`SummaryPolicy`、`plan_compaction`、`CompactionPlan.accept`、`acompact`、
  `Summary`、`covered_end`）；`Compaction` 放进 `state.py`，`ContextState.compaction` 随 JSON 存取；`fit()` 把应用摘要
  作为内部步骤、走同一个运行时检查；新错误 `EmptySummary`、`TruncatedSummary`。与本文的偏差：`created_at` 是 ISO
  字符串；`covered_count` 去掉（摘要值已经覆盖条数）；`plan.covered` 给出覆盖的 key。
  规则 → 测试（`tests/test_compaction.py`，除另注）：
  - 覆盖范围：`test_a_plan_covers_every_turn_but_the_kept_ones_and_renders_them`、
    `test_a_pinned_prefix_stays_out_and_a_later_pinned_message_stops_the_cut`、`test_nothing_to_cover_gives_no_plan`、
    `test_compaction_needs_keys`。
  - 后一次吃进前一次：`test_a_later_compaction_takes_in_the_earlier_summary`、
    `test_nothing_new_to_cover_after_a_compaction_gives_no_plan`。
  - 截断、空摘要失败，抄进来的头标去掉：`test_an_empty_or_truncated_summary_is_refused`、
    `test_header_lines_the_model_copied_are_removed`。
  - 应用摘要、报告、输入不变：`test_fit_shows_the_summary_in_place_of_what_it_covers_and_reports_it`。
  - 过期：`test_a_summary_that_no_longer_matches_the_history_is_stale`（删掉覆盖的最后一条、之前插入、改 key、
    不在轮边界）、`test_a_summary_with_keyless_messages_is_stale`。
  - 切点量摘要之后的历史、新摘要不带切点：`test_cuts_are_measured_after_the_summary_and_start_again_after_a_new_one`。
  - JSON：`test_state_with_a_compaction_survives_json`；`acompact`：`test_acompact_plans_calls_the_callers_model_and_accepts`。
  - 端到端：`tests/test_growth.py::test_a_growing_history_with_summaries_replays_stably`（60 个种子，需要时就写摘要，
    共写了 60 次；无异常、切点和摘要都没动时前缀不变）。
  见红：测试先写，没有实现时收集失败；之后破坏 10 处（摘要值、轮边界、截断、头标清理、停在 pinned、没有新内容、
  吃进前一次、`state` 保留摘要、无 key 检查、条数检查）：前 9 处有测试变红；条数检查破坏后没有测试变红，因为摘要值
  已经覆盖条数，于是删掉了条数这个字段；无 key 检查原来没有测试，补上并见红。

  收尾：`/simplify`（一个子代理，四个角度）与代码 review（Codex 不可用，子代理代替）同时跑。
  review 7 条：1、2、5 `end` 可能落到历史末尾（`keep_last_turns=0`、没有人话、负数），摘要盖住当前一轮或没完成的
  批次，之后的状态用不了 → `end` 永远是某一轮的开头，`keep_last_turns >= 1`
  （`test_keep_last_turns_must_keep_at_least_the_current_turn`、`test_a_history_without_human_turns_gives_no_plan`、
  `test_fewer_turns_than_kept_gives_no_plan`、`test_the_plan_never_covers_an_unfinished_batch_or_the_last_turn`）；
  3 `provenance` 不是 JSON 时存储时才失败、tuple 往返不等 → `accept` 里按 JSON 规整
  （`test_provenance_is_stored_as_json_gives_it_back`）；4 头标清理只认十六进制 mark → plan 记下 mark、只清这个
  （`test_copied_headers_are_removed_whatever_the_mark`）；7 前一次摘要带着前言交给模型 → 先去掉
  （`test_the_earlier_summary_goes_back_to_the_model_without_its_preamble`）；6 摘要后面是连续的 user 消息 → 写进
  §6.2。
  `/simplify` 8 条：采纳——两个相同的摘要函数合成 `digest`；`pinned_prefix` 移到 `structure.py`；`next()` 代替只取第一项
  的列表；`render` 按消息拆出 `_render`；`_summarised` 只收 `entries`，`SUMMARY_REF` 放到模块开头；`Compaction` 的 JSON
  用 `asdict` / `fields`；`__all__` 只列本模块定义的。跳过——`human_turns` 算两次（O(n)，便宜）；把 `SummaryPolicy`
  改成 Protocol（lite-agent 在 C7 可以继承或 `replace` 它，届时再定）。
  见红：回归测试先红后绿；修复处逐一破坏（`keep_last_turns` 校验、轮数不足、JSON 规整、按 mark 清理、去前言），
  每处都有测试变红；"轮数不足"一处起初没有测试变红，补了 `test_fewer_turns_than_kept_gives_no_plan`。

- **C2（2026-11-26）**：`docs.py`（`split`、`pack`、`halving`、`reduce_tree`、`select`，泛型）；`steps._longest`
  移到 `count.longest_prefix`（`split` 也用），按文本自己的字符/token 比定窗口再二分，1 MB 文本切 8k 段约 20 ms。与本文
  的偏差：一个模块而不是 `docs/` 包；`pack` 不切超大单元（单元的类型由调用方定，切不了），超了就 `DoesNotFit`；
  `reduce_tree` 返回能一起放进预算的笔记，最终 reduce 由调用方做；`select` 遇到第一条放不下的就停。
  规则 → 测试（`tests/test_docs.py`）：装得下原样一段 `test_text_that_fits_is_one_piece_exactly_as_given`；覆盖、
  重叠、切在换行 `test_pieces_cover_the_text_overlap_and_end_at_a_line_break`；没有换行也切得开
  `test_a_long_line_without_breaks_is_still_cut_to_size`；标题 `test_each_piece_carries_the_heading_above_where_it_starts`；
  默认估算下不超限、1 MB 速度 `test_split_respects_the_limit_with_the_default_estimate_and_cuts_a_large_text_quickly`；
  `pack` 两条；`halving` 三条（只拆太大的、别的失败一次调用后原样抛出、到最小仍抛出）；`reduce_tree` 五条（装得下不
  合并、按层按序合并、单条合不了失败、到深度限制失败、合并结果超预算失败）；`select` 一条（排序、去重、遇超即停）。
  见红：测试先写，没有实现时收集失败；之后破坏 13 处，起初 4 处没有测试变红：装得下的提前返回（与一般路径等价，
  删掉）、切在换行（测试文本恰好整行落在 500，改了行长）、只拆太大的（不拆也会最终抛出同一个异常，测试改成断言只调
  用一次）、深度限制（测试先撞上"没有进展"，加了一个每层都有进展的场景）；改完后都变红。

  收尾：`/simplify`（子代理，四个角度）与代码 review（Codex 不可用，子代理代替）同时跑，两边去重后：
  1. `halving` 的 `min_size=0` 无限递归 → 至少为 1（`test_halving_needs_a_minimum_of_at_least_one`）。
  2. 捕获 `BaseException` 把取消交给了调用方的判断函数 → 只捕获 `Exception`
     （`test_a_cancellation_never_reaches_the_size_predicate`）。
  3. `select` 停下之后会丢块、`over_budget` 里有重复 → 停下后照样去重，三个列表分完输入
     （`test_select_sorts_every_chunk_into_exactly_one_list`）。
  4. `split` 每段都把剩下的全文数一遍，真实分词器下是全文的 66 倍 → 按小段只数一次，约 3 倍
     （`test_split_counts_a_bounded_window_not_the_whole_rest_each_time`）；不可加的计数器有测试
     （`test_a_counter_that_charges_more_for_the_whole_than_its_parts_is_still_respected`）；计数器自相矛盾时报错而不死循环。
  5. 开在标题行上的段带的是上一个标题 → 带它自己所在章节的（`test_a_piece_that_starts_on_a_heading_carries_that_heading`，
     原来那条标题测试的期望随之改）。
  6. `Halved` 说不出哪些结果来自哪一段 → 每次调用一条记录（`test_halving_says_which_results_each_call_gave`）。
  7. 四处相同的 `DoesNotFit` 构造合成 `_too_big`；`reduce_tree` 每层每条只数一次；去掉死代码。`DoesNotFit` 带的
     `Report` 在这里只有 limit 和 overhead 有意义——没改成可选（会改公开接口），记在这里。
  见红：回归测试先红后绿；修复处逐一破坏，都有测试变红；"超长小段先截断"一处破坏后测试卡死，加了防护让它报错，
  之后破坏是一条测试失败而不是卡住。

- **C3（2026-11-26，准备）**：版本 0.1.0；CI 加 `package`（`uv build`，在干净的 3.11 环境里只装这个 wheel，用
  `python -I` 跑 `scripts/smoke.py`：README 的完整例子，adapter → `fit` → JSON 状态 → `acompact` → 再 `fit` → `split`，
  断言具体结果）和 `release`（v* 标签时把 `package` 构建的那份发布到 GitHub Release）。本地验证：uv 建的 3.11 环境、
  pip 建的 3.11 venv、一个 uv 3.12 新项目 `uv add <wheel>`，三处 smoke 都通过。README 改成真实用法和安装方式。
  发现：步骤列表先赋给变量时 mypy 推成 `list[object]`，要标注 `list[Step]`（README 写明）。
  发布（owner 同意）：`v0.1.0` 标签 → CI 四个 job 全绿（test 3.11、3.12，package，release）→ GitHub Release 带
  `lite_context-0.1.0-py3-none-any.whl` 和 sdist。验收：从 release URL 用 pip（3.11 venv）和 `uv add`（3.12 新项目）
  安装，smoke 都通过。

- **v0.1.1（2026-11-26，C4 准备时发现）**：wheel 里没有 `py.typed`，宿主的 pyright / mypy 会把 lite-context 当成
  无类型的包。加上 `src/lite_context/py.typed`；CI 的 `package` job 检查 wheel 里有它（加之前在本地构建确认这条检查
  失败）。

## 15. 待确认

- **厂商的 count_tokens**：宿主 B 对 Claude 用 SDK 的 `messages.count_tokens`（一次网络调用）。核心的 `Counter`
  是同步的；要不要加一个 async 计数器，或让宿主先算好再传 `Opaque.tokens` / 预算，C4 接入时定。
- **摘要质量怎么测**：masking 和摘要哪个更好，公开结果是 reported。C4 之后用 lite-eval 在宿主的真实任务上对比，
  测之前不在文档里推荐其中一个做默认。
- **服务端摘要**：REST 模式下由服务自己调 lite-model role 写摘要（§0 里"以后的可选项"）。有非 Python 宿主要用时再做。
- **token 级压缩**（LLMLingua 一类）：需要本地小模型，适合长文档不适合历史；有宿主要用时作为 §7 的可选步骤。
- **Python 3.11 的下限**：宿主 B 升到 3.12 后，是否提到 3.12 与其它 lite 模块一致。
- **项目 KB 的位置**：`KB/projects/lite-context/`（S0 创建）。
- **`pack` 要不要允许单个超预算的单元自成一组**（C4 宿主 A）：宿主 A 的按页分组让一页超预算时自成一组、交给模型
  （模型拒绝时再对半一次），`pack` 则直接 `DoesNotFit`。宿主 A 的病例流程这次不改（它的 §8.3 是有意的设计）；
  等有宿主需要时再加一个选项。
